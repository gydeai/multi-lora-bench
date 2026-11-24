"""Load test for an OpenAI-compatible server that serves many LoRA adapters (vLLM or anything else).

Sends streamed chat requests at a set concurrency, picking which adapter (model name) each request goes to,
and reports output tokens/s, time to first token and errors. Python 3.9+, standard library only.

Examples:
  python bench.py --base-url http://localhost:8000/v1 --models models.txt --pattern even --concurrency 32
  python bench.py --base-url http://localhost:8000/v1 --models models.txt --pattern skewed --concurrency 64
  python bench.py --base-url http://localhost:8000/v1 --models models.txt --pattern longdoc --concurrency 8

models.txt has one served model name per line (for vLLM, the names given to --lora-modules).
Set OPENAI_API_KEY if your server needs a key.
"""
import argparse, json, os, random, ssl, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

CHAT_PROMPTS = [
    "Explain in three sentences what a LoRA adapter is.",
    "Write a short haiku about the monsoon.",
    "List five uses of a hash map.",
    "Summarise the plot of a heist story in four lines.",
]

DOC = ("Invoice 4471 from Northwind Supplies to Contoso Ltd, dated 3 March. Line items: 40 units of part A-17 at $12.50, "
       "12 units of part B-02 at $88.00, freight $140.00, tax 18 percent. Payment terms net 30 to account 7712-0045. "
       "Notes: delivery delayed by two days due to a customs hold; the buyer asked for a partial credit. ")


def ssl_context():
    try:
        import certifi  # some Python installs have no CA bundle of their own
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def chat_body(max_tokens):
    return {"messages": [{"role": "user", "content": random.choice(CHAT_PROMPTS)}],
            "max_tokens": max_tokens, "temperature": 0.7}


def longdoc_body(max_tokens):
    # About 2,400 prompt tokens on average with the Qwen3 tokenizer. A random case number first, so the prefix cache can't reuse earlier prompts.
    doc = f"Case {random.randint(10**8, 10**9)}. " + " ".join(
        DOC.replace("4471", str(random.randint(1000, 9999))) for _ in range(random.randint(16, 26)))
    return {"messages": [{"role": "user", "content": doc + "\n\nExtract the vendor, total amount due and any dispute, as JSON."}],
            "max_tokens": max_tokens, "temperature": 0.2}


def call(base_url, key, model, body, ctx, extra):
    body = {**body, **extra, "model": model, "stream": True, "stream_options": {"include_usage": True}}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers=headers)
    t0, ttft, out_toks, in_toks = time.time(), None, 0, 0
    try:
        with urllib.request.urlopen(req, timeout=300, context=ctx) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                delta = (ev.get("choices") or [{}])[0].get("delta") or {}
                if ttft is None and (delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")):
                    ttft = time.time() - t0
                if ev.get("usage"):
                    out_toks = ev["usage"].get("completion_tokens", 0)
                    in_toks = ev["usage"].get("prompt_tokens", 0)
        return {"ok": True, "ttft": ttft, "latency": time.time() - t0, "out": out_toks, "in": in_toks}
    except urllib.error.HTTPError as e:
        return {"ok": False, "code": e.code, "body": e.read().decode()[:200]}
    except Exception as e:
        return {"ok": False, "code": "error", "body": repr(e)[:200]}


def pct(xs, p):
    return round(sorted(xs)[min(len(xs) - 1, int(p * len(xs)))], 2) if xs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="for example http://localhost:8000/v1")
    ap.add_argument("--models", required=True, help="file with one model name per line")
    ap.add_argument("--pattern", choices=["even", "skewed", "single", "longdoc"], default="even",
                    help="even: any adapter at random; skewed: 80%% of calls on the first 5; single: the first only; "
                         "longdoc: skewed, with long documents and short answers")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--requests", type=int, help="default: 6 x concurrency")
    ap.add_argument("--max-tokens", type=int, help="default: 200 for chat, 150 for longdoc")
    ap.add_argument("--no-thinking", action="store_true",
                    help="send chat_template_kwargs enable_thinking=false (Qwen3 and similar)")
    ap.add_argument("--out", help="also write the result as JSON to this file")
    a = ap.parse_args()

    models = [m.strip() for m in open(a.models) if m.strip() and not m.startswith("#")]
    hot, cold = models[:5], models[5:] or models[:5]
    pick = {"even": lambda: random.choice(models),
            "skewed": lambda: random.choice(hot) if random.random() < 0.8 else random.choice(cold),
            "single": lambda: models[0]}
    pick["longdoc"] = pick["skewed"]
    make = longdoc_body if a.pattern == "longdoc" else chat_body
    max_tokens = a.max_tokens or (150 if a.pattern == "longdoc" else 200)
    n = a.requests or a.concurrency * 6
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if a.no_thinking else {}
    key, ctx = os.environ.get("OPENAI_API_KEY"), ssl_context()

    jobs = [pick[a.pattern]() for _ in range(n)]
    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(lambda m: call(a.base_url, key, m, make(max_tokens), ctx, extra), jobs))
    wall = time.time() - t0

    ok = [r for r in res if r["ok"]]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    out_toks, in_toks = sum(r["out"] for r in ok), sum(r["in"] for r in ok)
    result = {
        "pattern": a.pattern, "concurrency": a.concurrency, "requests": n, "ok": len(ok), "errors": n - len(ok),
        "adapters": len(models), "adapters_hit": len(set(jobs)), "wall_s": round(wall, 1),
        "output_tok_per_s": round(out_toks / wall, 1), "input_tok_per_s": round(in_toks / wall, 1),
        "ttft_p50_s": pct(ttfts, .5), "ttft_p95_s": pct(ttfts, .95),
        "latency_p50_s": pct([r["latency"] for r in ok], .5),
        "error_samples": [{"code": r["code"], "body": r["body"]} for r in res if not r["ok"]][:3],
    }
    print(json.dumps(result, indent=1))
    if a.out:
        json.dump(result, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
