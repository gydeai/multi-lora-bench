# How many LoRA adapters can one small GPU serve?

We put 39 public Qwen3-8B LoRA adapters from Hugging Face on one 24 GB NVIDIA A30, all at the same time, and sent traffic at them with vLLM. This repo has the adapters, the exact vLLM command, the load-test script and the raw results, so you can check our numbers or run the same test on your own GPU.

Short version: **adapter swapping, not the GPU, sets your throughput.** At 32 requests at a time, one adapter alone did 806 output tokens/s. The same load spread evenly over 39 adapters did 231.

## Setup

- GPU: 1 x NVIDIA A30, 24 GB
- vLLM 0.33.0, Qwen3-8B in bf16 ([revision b968826](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218)), `--max-model-len 8192 --gpu-memory-utilization 0.92`
- LoRA: `--enable-lora --max-lora-rank 64 --max-loras 8 --max-cpu-loras 32` (see [serve.sh](serve.sh))
- Adapters: the newest public Qwen3-8B adapter from each of 40 different authors on Hugging Face, ranks 8 to 64. 39 loaded. One failed because its tensor names came from a different model class. The list, with ranks and exact revisions, is in [adapters.json](adapters.json).
- Traffic: streamed chat requests, Qwen3 thinking off. Chat: short questions, up to 200 output tokens. Long documents: about 2,400 prompt tokens on average, up to 150 output tokens (about 50 used). Script: [bench.py](bench.py).

**Read this before comparing numbers.** We measured through our own serving stack (TLS, a gateway with API keys and usage counting, then vLLM), not against vLLM directly. Raw vLLM on the same GPU should be a little faster, so treat these as a floor. Every run in the tables had 0 errors. Two earlier runs had errors from our gateway's per-key rate limit, not from vLLM; we raised the limit and ran them again. Only the clean runs are here.

## Results

### Chat, 8 GPU adapter slots (max rank 64), 39 adapters

Output tokens/s, then median time to first token.

| Traffic | 8 at a time | 32 at a time | 64 at a time |
|---|---|---|---|
| Spread evenly over 39 adapters | 153 · 0.25 s | 231 · 6.1 s | |
| 80% of calls on 5 adapters | | 418 · 0.28 s | 541 · 0.30 s |
| One adapter only | | 806 · 0.22 s | 1,390 · 0.25 s |

### Chat, 4 slots vs 8 slots, spread evenly over 39 adapters

| Slots | 8 at a time | 32 at a time |
|---|---|---|
| `--max-loras 4` | 114 · 2.3 s | 159 · 14.1 s |
| `--max-loras 8` | 153 · 0.25 s | 231 · 6.1 s |

### Chat, 16 slots (max rank 32), 36 adapters

At max rank 32, vLLM refuses the three rank-64 adapters, so these runs used 36.

| Traffic | 32 at a time | 64 at a time |
|---|---|---|
| Spread evenly over 36 adapters | 394 · 0.76 s | 581 · 0.92 s |
| 80% of calls on 5 adapters | 601 · 0.26 s | 864 · 0.31 s |

16 slots at max rank 64 did not fit: vLLM refused to start ("No available memory for the cache blocks"). 12 slots did not fit either. At 8 slots and rank 64, the KV cache was 13,760 tokens.

### Long documents, 8 slots, 80% of calls on 5 adapters

| At a time | Prompt tokens/s | Output tokens/s | Median time to first token |
|---|---|---|---|
| 8 | 2,927 | 64 | 3.0 s |
| 32 | 3,155 | 61 | 21.5 s |

Raw numbers for every table are in [results/](results/).

## What we learned

1. **Adapter swapping decides your speed, not the GPU.** At 32 requests at a time, one adapter: 806 tokens/s. 39 adapters spread evenly: 231. Same GPU, same model.
2. **`--max-loras` is the setting that matters.** It is how many adapters sit on the GPU at once. Going from 4 to 8 cut time to first token at 8 requests at a time from 2.3 s to 0.25 s.
3. **Slots cost memory.** Each slot reserves room for an adapter of `--max-lora-rank`, taken from the KV cache. At rank 64, 16 slots didn't fit in 24 GB. At rank 32 they did, and throughput rose a lot. The catch: about 1 in 10 recent Qwen3-8B adapters on the Hub use a rank above 32 (18 of 178 we checked), and vLLM refuses them.
4. **Real traffic is lopsided, and that helps.** With 80% of calls on 5 adapters, throughput nearly doubled compared with an even spread (418 vs 231 tokens/s at 32 requests at a time).
5. **Long prompts are cheap.** The A30 read about 3,000 prompt tokens a second. Extraction and classification work suits shared serving well; long chat answers are the hardest case.
6. **Qwen3 thinking:** add `--reasoning-parser qwen3`, or the `<think>` text ends up inside the reply.
7. **Restarts take a while.** In our setup a vLLM restart took the endpoint down for 132 to 168 seconds.

## Run it yourself

Needs a GPU with about 24 GB, vLLM 0.33 (`pip install vllm==0.33.0`) and `huggingface_hub`.

```bash
./download.sh                 # the 39 adapters at the tested revisions, and models.txt
./serve.sh                    # vLLM with 8 slots, max rank 64
python bench.py --base-url http://localhost:8000/v1 --models models.txt --no-thinking --pattern even --concurrency 32
python bench.py --base-url http://localhost:8000/v1 --models models.txt --no-thinking --pattern skewed --concurrency 64
python bench.py --base-url http://localhost:8000/v1 --models models.txt --no-thinking --pattern single --concurrency 64
python bench.py --base-url http://localhost:8000/v1 --models models.txt --no-thinking --pattern longdoc --concurrency 8
```

For 16 slots: `MAX_RANK=32 ./download.sh`, then `SLOTS=16 RANK=32 ./serve.sh`.

Here vLLM loads the adapters from disk at startup with `--lora-modules`. In our setup they were added at runtime through `/v1/load_lora_adapter`; the serving path is the same once they are loaded.

If you run it on other GPUs (L4, A100, H100, 4090), we'd like to see your numbers. Open an issue or a pull request with your results file.

## Who we are

We build [Portweights](https://portweights.com/?ref=gh-bench), which runs this setup as a service: give it a LoRA adapter, get a private OpenAI-compatible API, paid per token. This test is how we chose our settings. Everything here works without Portweights.

## License

MIT. The adapters belong to their authors, under their own licenses; this repo only lists them.
