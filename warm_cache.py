# warm_cache.py -- run from the project root:  python warm_cache.py
#
# Pre-computes answers overnight so that the questions students (and your guide) are likely to ask
# come back from the cache in seconds instead of minutes.
#
# It reads warm_questions.txt, one question per line:
#       question                       (Auto-detect subject, like the app's default)
#       DS LAB | write a program for binary search
# Lines starting with # are ignored. It asks each question exactly the way the chat tab does
# (5-mark depth, fresh conversation) and the answers land in _answer_cache.json.
# Safe to stop (Ctrl+C) and run again: finished questions are served from the cache, so they take
# seconds. Keep Ollama running and the laptop plugged in. Do NOT use the app at the same time.
#
# Note: a cached answer is reused only when the question, subject and retrieved notes are the same.
# Use the same subject selection here that you will use when asking.

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ADMIN_PASSWORD", "warm-cache-run")  # keeps config.py quiet
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

if not getattr(config, "ANSWER_CACHE", True):
    sys.exit("ANSWER_CACHE is False in config.py, so nothing would be saved. "
             "Set ANSWER_CACHE = True, save, and run this again.")


def cache_size():
    try:
        return len(json.loads((Path(config.FAISS_DIR) / "_answer_cache.json").read_text(encoding="utf-8")))
    except Exception:
        return 0

try:
    from src.rag_pipeline import RAGPipeline
except ImportError:
    from rag_pipeline import RAGPipeline

MARKS = 5  # DEFAULT_CHAT_MARKS in app.py


def read_questions(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            subj, q = [p.strip() for p in line.split("|", 1)]
        else:
            subj, q = "", line
        out.append((subj or None, q))
    return out


def main():
    qfile = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "warm_questions.txt"
    items = read_questions(qfile)
    print(f"{len(items)} questions to warm from {qfile.name}. Loading the pipeline...")
    rag = RAGPipeline()
    done = failed = 0
    t_all = time.time()
    for n, (subj, q) in enumerate(items, 1):
        t0 = time.time()
        try:
            for _kind, _payload in rag.ask_stream(q, MARKS, subject=subj, context_topic=None):
                pass
            done += 1
            status = "ok"
        except Exception as e:
            failed += 1
            status = f"FAILED ({e})"
        size = cache_size()
        print(f"[{n}/{len(items)}] {time.time() - t0:5.0f}s  {status}  cache={size}  {(subj or 'Auto')} | {q[:70]}", flush=True)
        if n == 1 and size == 0 and status == "ok":
            sys.exit("The first answer was NOT saved to faiss_store\\_answer_cache.json (file is empty or "
                     "unwritable). Stopping so you do not waste hours. Check the folder is writable and that "
                     "no other program has the file open, then run this again.")
    print(f"\nFinished: {done} ok, {failed} failed, {(time.time() - t_all) / 60:.0f} minutes.")


if __name__ == "__main__":
    main()