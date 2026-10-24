# calibrate_cutoff.py -- run from the project root:  python calibrate_cutoff.py
#
# Measures how well the question-matching distance cutoff (currently 1.15 in
# question_matcher.py) separates RIGHT matches from WRONG ones, using the answer key
# in calibration_set.csv. It does NOT call the LLM, so it takes about a minute.
# Ollama must be running (it is only used to embed the test questions with nomic-embed-text).
#
# calibration_set.csv columns:
#   query     what a student would type
#   subject   subject code as in the question bank (DS, DBMS, OS...) or empty for "any subject"
#   keywords  words/phrases (separated by |) that a RIGHT paper question must contain,
#             or NONE when nothing in the papers should match
#
# A paper question counts as RIGHT for a row when it is in that subject (if given) and contains
# one of the keywords. Keyword rules are the same as in question_matcher.py.
#
# Outputs:  calibration_results.csv (one line per cutoff)
#           calibration_detail.csv  (what each query actually returned -- read this!)

import os
import re
import csv
import sys
import json
import argparse
from pathlib import Path

os.environ.setdefault("ADMIN_PASSWORD", "calibration-run")  # keeps config.py quiet

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def kw_match(keyword: str, text: str) -> bool:
    """Same word-boundary rule as question_matcher._topic_in_text (text must be lower-case)."""
    kw = keyword.strip().lower()
    if not kw:
        return False
    if len(kw) <= 3:
        pat = r"(?<![a-z0-9])" + re.escape(kw) + r"s?(?![a-z0-9])"
    else:
        pat = r"(?<![a-z0-9])" + re.escape(kw)
    return re.search(pat, text) is not None


def load_everything(embeddings=None):
    from config import FAISS_DIR
    faiss_dir = Path(FAISS_DIR)
    if not (faiss_dir / "_question_bank.json").exists():
        faiss_dir = HERE / "faiss_store"
    bank = json.load(open(faiss_dir / "_question_bank.json", encoding="utf-8"))
    if embeddings is None:
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://127.0.0.1:11434")
    from langchain_community.vectorstores import FAISS
    store = FAISS.load_local(str(faiss_dir / "_question_bank_faiss"), embeddings,
                             allow_dangerous_deserialization=True)
    return bank, store


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            q = (r.get("query") or "").strip()
            if not q:
                continue
            rows.append({
                "query": q,
                "subject": (r.get("subject") or "").strip().upper(),
                "keywords": [k.strip() for k in (r.get("keywords") or "").split("|") if k.strip()],
            })
    return rows


def retrieve(store, bank, row, k):
    """Mirror the pipeline: top-k from the whole question store, THEN filter by subject."""
    out = []
    for doc, dist in store.similarity_search_with_score(row["query"], k=k):
        rid = (doc.metadata or {}).get("record_id")
        if rid is None or not (0 <= rid < len(bank)):
            continue
        if row["subject"] and bank[rid]["subject"].strip().upper() != row["subject"]:
            continue
        out.append((rid, float(dist)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "calibration_set.csv"))
    ap.add_argument("--k", type=int, default=25, help="candidates fetched per query (pipeline uses 25)")
    ap.add_argument("--current", type=float, default=1.15, help="cutoff currently used")
    ap.add_argument("--embeddings", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    bank, store = load_everything()
    rows = read_rows(args.csv)
    if not rows:
        sys.exit("No rows found in " + args.csv)

    # ---- ground truth ----
    for r in rows:
        r["none"] = (len(r["keywords"]) == 1 and r["keywords"][0].upper() == "NONE")
        r["relevant"] = set()
        if not r["none"]:
            for i, q in enumerate(bank):
                if r["subject"] and q["subject"].strip().upper() != r["subject"]:
                    continue
                text = q["question"].lower()
                if any(kw_match(k, text) for k in r["keywords"]):
                    r["relevant"].add(i)
            if not r["relevant"]:
                print(f"WARNING: no paper question contains {r['keywords']} for '{r['query']}' "
                      f"(subject {r['subject'] or 'any'}) -- fix the keywords or mark it NONE")
        r["cands"] = retrieve(store, bank, r, args.k)

    pos = [r for r in rows if not r["none"] and r["relevant"]]
    neg = [r for r in rows if r["none"]]
    print(f"\n{len(pos)} queries that should match, {len(neg)} that should not.\n")

    # ---- sweep cutoffs ----
    maxd = max([d for r in rows for _, d in r["cands"]] + [1.0])
    cutoffs = []
    c = 0.30
    while c <= min(maxd + 0.05, 2.0) + 1e-9:
        cutoffs.append(round(c, 2))
        c += 0.05
    for extra in (args.current,):
        if extra not in cutoffs:
            cutoffs.append(extra)
    cutoffs.sort()

    total_relevant = sum(len(r["relevant"]) for r in pos)
    results = []
    for c in cutoffs:
        tp = fp = 0
        queries_hit = 0
        for r in pos:
            ret = [rid for rid, d in r["cands"] if d <= c]
            good = [rid for rid in ret if rid in r["relevant"]]
            tp += len(good)
            fp += len(ret) - len(good)
            if good:
                queries_hit += 1
        false_alarm = sum(1 for r in neg if any(d <= c for _, d in r["cands"]))
        # false-alarm rows also add their returned questions to fp
        fp_neg = sum(1 for r in neg for _, d in r["cands"] if d <= c)
        prec = tp / (tp + fp + fp_neg) if (tp + fp + fp_neg) else 0.0
        rec = tp / total_relevant if total_relevant else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        results.append({
            "cutoff": c, "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "queries_with_a_correct_hit": f"{queries_hit}/{len(pos)}",
            "nothing_queries_wrongly_answered": f"{false_alarm}/{len(neg)}",
            "wrong_questions_shown": fp + fp_neg,
        })

    best = max(results, key=lambda x: (x["f1"], -x["cutoff"]))
    hdr = f"{'cutoff':>7} {'precision':>10} {'recall':>7} {'F1':>6} {'queries hit':>12} {'false alarms':>13} {'wrong shown':>12}"
    print(hdr)
    print("-" * len(hdr))
    for x in results:
        mark = "  <- best F1" if x is best else ("  <- current" if abs(x["cutoff"] - args.current) < 1e-9 else "")
        print(f"{x['cutoff']:>7.2f} {x['precision']:>10.3f} {x['recall']:>7.3f} {x['f1']:>6.3f} "
              f"{x['queries_with_a_correct_hit']:>12} {x['nothing_queries_wrongly_answered']:>13} "
              f"{x['wrong_questions_shown']:>12}{mark}")

    with open(HERE / "calibration_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # ---- per-query detail ----
    with open(HERE / "calibration_detail.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "subject", "should_match", "best_distance", "nearest_question",
                    "nearest_is_right", "rank_of_first_right", "distance_of_first_right"])
        for r in rows:
            cands = sorted(r["cands"], key=lambda t: t[1])
            if cands:
                rid, d = cands[0]
                nearest = bank[rid]["question"][:120].replace("\n", " ")
                right = "" if r["none"] else str(rid in r["relevant"])
            else:
                d, nearest, right = "", "(nothing retrieved)", ""
            first_rank, first_d = "", ""
            if not r["none"]:
                for n, (rid, dd) in enumerate(cands, 1):
                    if rid in r["relevant"]:
                        first_rank, first_d = n, round(dd, 3)
                        break
            w.writerow([r["query"], r["subject"], "NO" if r["none"] else "YES",
                        round(d, 3) if d != "" else "", nearest, right, first_rank, first_d])

    cur = next((x for x in results if abs(x["cutoff"] - args.current) < 1e-9), None)
    print(f"\nBest cutoff on this set: {best['cutoff']} (precision {best['precision']}, recall {best['recall']}).")
    if cur:
        print(f"Current cutoff {args.current}: precision {cur['precision']}, recall {cur['recall']}.")
    print("\nWrote calibration_results.csv and calibration_detail.csv.")
    print("Open calibration_detail.csv: rows where nearest_is_right is False deserve a manual look --")
    print("the nearest question may still be a sensible match your keywords did not cover.")


if __name__ == "__main__":
    main()
