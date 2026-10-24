# eval_questions.py
"""
uniRAG regression + calibration harness.

Runs a fixed set of questions through the SAME path the chat tab uses
(RAGPipeline.ask_stream) and checks each answer with simple, explicit rules,
so a fix can be verified — and shown not to break anything else — in one run
instead of by eye.

Run from the project root (next to app.py), with Ollama running:

    python eval_questions.py                 # everything
    python eval_questions.py --only bleed    # cases whose name contains "bleed"
    python eval_questions.py --calibrate     # only print retrieval distances
    python eval_questions.py --list          # list case names

It prints PASS / FAIL / WARN per check and writes the complete answers to
eval_report.txt — send that file if you want me to look at the details.

FAIL  = a rule the system is supposed to guarantee was broken.
WARN  = a known model-quality risk (e.g. a wrong fact) that code cannot prove or
        prevent; worth reading by eye.
"""
import argparse
import re
import sys
import time

ALL = "All / Auto-detect"


# ---------------------------------------------------------------------------
# Rule helpers. Each returns (ok: bool, detail: str)
# ---------------------------------------------------------------------------
def contains(*needles):
    def rule(ans, ctx):
        low = ans.lower()
        missing = [n for n in needles if n.lower() not in low]
        return (not missing, f"missing {missing}" if missing else "")
    rule.label = f"contains {list(needles)}"
    return rule


def contains_any(*needles):
    def rule(ans, ctx):
        low = ans.lower()
        ok = any(n.lower() in low for n in needles)
        return (ok, "" if ok else f"none of {list(needles)} found")
    rule.label = f"contains any of {list(needles)}"
    return rule


def not_contains(*needles):
    def rule(ans, ctx):
        low = ans.lower()
        found = [n for n in needles if n.lower() in low]
        return (not found, f"unexpectedly found {found}" if found else "")
    rule.label = f"does not contain {list(needles)}"
    return rule


def no_regex(pattern, label):
    def rule(ans, ctx):
        m = re.search(pattern, ans, re.IGNORECASE)
        return (m is None, f"found {m.group(0)!r}" if m else "")
    rule.label = label
    return rule


def intent_is(*names):
    """The query was classified with these intent(s) (REVIEW_FIXES_APPLIED)."""
    def rule(ans, ctx):
        dbg = (ctx.get("payload") or {}).get("debug_info") or {}
        got = dbg.get("intents") or dbg.get("intent") or []
        if isinstance(got, str):
            got = [got]
        ok = all(n in got for n in names)
        return (ok, "" if ok else f"intent was {got}, expected {list(names)}")
    rule.label = f"intent is {list(names)}"
    return rule


def sittings_match_years(ans, ctx):
    """'Found across N examination sitting(s)' must equal the number of years listed."""
    sit = re.search(r"across\s+\**(\d+)\**\s+examination sitting", ans, re.IGNORECASE)
    yrs = re.search(r"Exam Years:\**\s*([^\n]+)", ans, re.IGNORECASE)
    if not sit or not yrs:
        return (True, "n/a")
    listed = [y for y in re.findall(r"\b(?:19|20)\d{2}\b", yrs.group(1))]
    ok = int(sit.group(1)) == len(set(listed))
    return (ok, "" if ok else f"says {sit.group(1)} sittings but lists {sorted(set(listed))}")
sittings_match_years.label = "sitting count equals years listed"


def worst_case_nlogn(ans, ctx):
    """Quick sort's worst case is O(n^2). Flags 'worst' near 'n log n'."""
    for sent in re.split(r"(?<=[.!?])\s+|\n+", ans):
        low = sent.lower()
        if ("worst" in low and re.search(r"n\s*log\s*n", low)
                and not re.search(r"n\s*\^\s*2|n2|n\u00b2|quadratic", low)):
            return (False, "a sentence pairs 'worst case' with O(n log n) and not O(n^2)")
    return (True, "")
worst_case_nlogn.label = "quick sort worst case not stated as n log n"


def code_is_c(ans, ctx):
    # Regexes anchored to line starts: a plain "def " substring also matches "typedef struct".
    py_or_java = [
        r"(?m)^\s*def\s+\w+\s*\(", r"(?m)^\s*print\(", r"(?m)^\s*import\s+\w+\s*$",
        r"public\s+static", r"System\.out", r"import\s+java",
    ]
    bad = [p for p in py_or_java if re.search(p, ans)]
    return ("#include" in ans and not bad, f"not C code (matched {bad})" if bad else ("no #include" if "#include" not in ans else ""))
code_is_c.label = "program is written in C"


def same_as_first_run(ans, ctx):
    first = ctx.get("first_answer")
    return (first is not None and first == ans, "answer changed between two identical runs")
same_as_first_run.label = "identical question gives identical answer"


# ---------------------------------------------------------------------------
# Test cases: (name, subject, [questions...], {index_of_question: [rules]}, severity)
# The LAST question in a case is what the rules check unless an index is given.
# Earlier questions build conversation context, exactly like a real chat.
# ---------------------------------------------------------------------------
NO_FIG = no_regex(r"\b(?:Fig(?:ure|\.)?|Table)\s*\d+|§", "no figure/table/section references")

CASES = [
    dict(name="definition_basic", subject=ALL, questions=["what is decision tree"],
         rules=[contains("decision tree"), NO_FIG]),
    dict(name="definition_typos", subject=ALL, questions=["wat is normalisation in dbms"],
         rules=[contains("normali"), NO_FIG]),
    # Either the out-of-syllabus gate (checked first, for plain "what is X" questions)
    # or the separate notes-grounding check can be the one that refuses these — both are
    # correct, valid reasons, so both wordings are accepted here.
    dict(name="ungrounded_fenwick", subject=ALL, questions=["what is a Fenwick tree"],
         rules=[contains_any("couldn't find", "don't have", "doesn't appear in your MCA syllabus")]),
    dict(name="ungrounded_kubernetes", subject=ALL, questions=["explain Kubernetes"],
         rules=[contains_any("couldn't find", "don't have", "doesn't appear in your MCA syllabus")]),
    # "health" is NOT a real out-of-syllabus case: the crypto book in this corpus covers RNG
    # "health testing" (NIST SP 800-90B), a real technical term, so answering it is CORRECT.
    # Kept here only as a documented, known non-failure — expect this one to legitimately
    # answer rather than refuse; do not "fix" the gate to block it.
    dict(name="known_ok_health_is_a_real_crypto_term", subject=ALL, questions=["what is health"],
         rules=[not_contains("doesn't appear in your MCA syllabus")]),  # the syllabus gate must not block it
    dict(name="out_of_syllabus_unambiguous", subject=ALL, questions=["explain the French revolution"],
         rules=[contains_any("doesn't appear in your MCA syllabus", "couldn't find", "don't have")]),
    # The gate must never block real syllabus topics (spelling variants, plurals, other subjects):
    dict(name="in_scope_quicksort", subject=ALL, questions=["what is quick sort"],
         rules=[not_contains("doesn't appear in your MCA syllabus", "couldn't find")]),
    dict(name="in_scope_avl", subject=ALL, questions=["what is an AVL tree"],
         rules=[not_contains("doesn't appear in your MCA syllabus", "couldn't find")]),
    dict(name="in_scope_scheduling", subject=ALL, questions=["what is process scheduling"],
         rules=[not_contains("doesn't appear in your MCA syllabus", "couldn't find")]),
    dict(name="in_scope_stack", subject=ALL, questions=["what is a stack"],
         rules=[not_contains("doesn't appear in your MCA syllabus", "couldn't find")]),
    dict(name="comparison_both_sides", subject=ALL, questions=["array vs linked list"],
         rules=[contains("array", "linked list"), contains("vs"), NO_FIG]),
    dict(name="comparison_stack_queue", subject=ALL, questions=["difference between stack and queue"],
         rules=[contains("stack", "queue"), NO_FIG]),
    dict(name="chain_followup_kept", subject=ALL,
         questions=["explain hashing", "what about its types", "how many times has that appeared"],
         rules=[contains("hashing"), contains("EXAM FREQUENCY")]),
    dict(name="bleed_subjects", subject=ALL,
         questions=["explain hashing", "how many subjects in 1st sem"],
         rules=[not_contains("hashing")]),
    dict(name="bleed_never_asked", subject=ALL,
         questions=["array vs linked list", "which topics have never been asked in exams"],
         rules=[not_contains("EXAM TREND: Array", "EXAM FREQUENCY: Array")]),
    dict(name="bleed_rdbms", subject=ALL,
         questions=["how many times has database redundancy elimination appeared", "has RDBMS come up"],
         rules=[not_contains("Normalization")]),
    dict(name="frequency_consistent_quicksort", subject=ALL,
         questions=["how many times has quick sort appeared"],
         rules=[contains("EXAM FREQUENCY"), sittings_match_years]),
    dict(name="frequency_consistent_queue", subject=ALL,
         questions=["what is queue and how many times has it appeared"],
         rules=[contains("EXAM HISTORY"), sittings_match_years]),
    dict(name="hinglish_frequency", subject=ALL,
         questions=["queue kitni baar aaya hai exam me"],
         rules=[contains("EXAM FREQUENCY", "queue")]),
    dict(name="exact_questions_clean", subject=ALL,
         questions=["show all 10-mark questions on normalization"],
         rules=[intent_is("exact_questions"), contains("EXAM HISTORY"), contains("normalization"),
                no_regex(r"Scanned with|P\.T\.O|F\.T\.O", "no scanner/page-turn noise")]),
    dict(name="ranking_scoped", subject="DS",
         questions=["what is the most frequently asked question in data structures"],
         rules=[contains("MOST FREQUENTLY"), not_contains("No repeated question clusters")]),
    dict(name="ranking_other_phrasing", subject="DS",
         questions=["rank the top 5 topics likely to appear in the exam"],
         rules=[contains("MOST FREQUENTLY")]),
    dict(name="trend_vs_frequency_same_topic", subject=ALL,
         questions=["has binary search tree been asked every year or only sometimes"],
         rules=[contains("EXAM TREND")]),
    dict(name="lab_binary_search", subject=ALL,
         questions=["write a program for binary search"],
         rules=[code_is_c, contains("mid"), not_contains("struct Node")]),
    dict(name="lab_linked_list", subject=ALL,
         questions=["write a program to insert and delete elements in a singly linked list"],
         rules=[code_is_c]),
    dict(name="diagram_request_honest", subject=ALL,
         questions=["draw the diagram of AVL tree rotation"],
         rules=[contains("aren't available")]),
    # CURRICULUM_PATCH_APPLIED: programme-structure questions are answered from the folders, not exam stats
    dict(name="curriculum_count", subject=ALL, questions=["how many subjects in 1st sem mca"],
         rules=[intent_is("curriculum"), contains("Semester 1"), not_contains("EXAM FREQUENCY")]),
    dict(name="curriculum_list", subject=ALL, questions=["what are the subjects in 3rd sem"],
         rules=[intent_is("curriculum"), contains("Semester 3"), not_contains("EXAM FREQUENCY")]),
    dict(name="curriculum_labs", subject=ALL, questions=["which lab in 2nd sem"],
         rules=[intent_is("curriculum"), contains("Semester 2")]),
    dict(name="invalid_input", subject=ALL, questions=["?"],
         rules=[contains("didn't catch")]),
    dict(name="known_risk_quicksort_complexity", subject=ALL,
         questions=["what is the time complexity of quick sort"],
         rules=[worst_case_nlogn], severity="WARN"),
    dict(name="determinism_stack_2023", subject=ALL, repeat_twice=True,
         questions=["show the exact question on stacks in 2023"],
         rules=[intent_is("exact_questions"), same_as_first_run]),
]

# Topics for --calibrate: distance for topics that ARE in the notes vs ones that are not.
IN_NOTES = ["decision tree", "hashing", "queue", "linked list", "normalization",
            "binary search tree", "recursion", "quick sort"]
NOT_IN_NOTES = ["fenwick tree", "kubernetes", "football", "health", "cooking recipe", "blockchain"]


# ---------------------------------------------------------------------------
def ask_like_chat(rag, question, subject, last_topic):
    """Consumes ask_stream exactly like the Streamlit chat tab does."""
    payload = {}
    for kind, data in rag.ask_stream(question, 8, subject=subject, context_topic=last_topic):
        if kind == "done":
            payload = data
    return payload


def run_case(rag, case):
    results, transcript = [], []
    last_topic, last_payload, first_answer = None, {}, None
    passes = 2 if case.get("repeat_twice") else 1
    for run in range(passes):
        last_topic = None
        for q in case["questions"]:
            t0 = time.time()
            last_payload = ask_like_chat(rag, q, case["subject"], last_topic)
            last_topic = last_payload.get("topic") or last_topic
            dbg = last_payload.get("debug_info") or {}
            transcript.append(
                f"[run {run + 1}] Q: {q}\n  intents={dbg.get('intents') or dbg.get('intent')} "
                f"topic={last_payload.get('topic')!r} best_distance={dbg.get('best_distance')} "
                f"({time.time() - t0:.0f}s)\n{last_payload.get('answer', '')}\n"
            )
        if run == 0:
            first_answer = last_payload.get("answer", "")
    answer = last_payload.get("answer", "")
    ctx = {"first_answer": first_answer, "payload": last_payload}
    severity = case.get("severity", "FAIL")
    for rule in case["rules"]:
        try:
            ok, detail = rule(answer, ctx)
        except Exception as e:  # a broken rule must never hide a real result
            ok, detail = False, f"rule crashed: {e}"
        results.append((ok, getattr(rule, "label", rule.__name__), detail, severity))
    return results, "\n".join(transcript)


def run_suite(rag, cases, report_path="eval_report.txt"):
    total_fail = total_warn = total_pass = 0
    report = []
    for case in cases:
        print(f"\n▶ {case['name']}   ({' → '.join(case['questions'])})")
        results, transcript = run_case(rag, case)
        report.append(f"===== {case['name']} =====\n{transcript}")
        for ok, label, detail, severity in results:
            if ok:
                total_pass += 1
                print(f"   PASS  {label}")
            elif severity == "WARN":
                total_warn += 1
                print(f"   WARN  {label}  — {detail}")
            else:
                total_fail += 1
                print(f"   FAIL  {label}  — {detail}")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(report))
    print(f"\n{'=' * 60}\nPASS {total_pass}   FAIL {total_fail}   WARN {total_warn}")
    print(f"Full answers written to {report_path}")
    return total_fail


def calibrate(rag):
    """Prints the best retrieval distance for topics that are / are not in the notes,
    so the relevance cutoff can be chosen from data instead of guessed."""
    print(f"\n{'topic':24} {'in notes?':10} {'best_distance':14} outcome")
    rows = []
    for expected, topics in (("yes", IN_NOTES), ("no", NOT_IN_NOTES)):
        for t in topics:
            p = ask_like_chat(rag, f"what is {t}", ALL, None)
            d = (p.get("debug_info") or {}).get("best_distance")
            refused = bool(re.search(r"couldn't find|don't have", p.get("answer", ""), re.I))
            rows.append((t, expected, d, refused))
            print(f"{t:24} {expected:10} {str(d):14} {'refused' if refused else 'answered'}")
    yes = [d for _, e, d, _ in rows if e == "yes" and d is not None]
    no = [d for _, e, d, _ in rows if e == "no" and d is not None]
    if yes and no:
        print(f"\nin-notes  distances: min {min(yes):.3f}  max {max(yes):.3f}")
        print(f"not-in-notes distances: min {min(no):.3f}  max {max(no):.3f}")
        if max(yes) < min(no):
            print(f"→ a cutoff anywhere between {max(yes):.3f} and {min(no):.3f} separates them cleanly.")
        else:
            print("→ the two ranges overlap: distance alone cannot separate them; the grounding check is doing the work.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run only cases whose name contains this text")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for c in CASES:
            print(c["name"])
        return 0

    from src.rag_pipeline import RAGPipeline  # imported late so --list needs no Ollama
    print("Loading pipeline (indexes + question bank)...")
    rag = RAGPipeline()

    if args.calibrate:
        calibrate(rag)
        return 0
    cases = [c for c in CASES if not args.only or args.only.lower() in c["name"].lower()]
    if not cases:
        print("No case matches that filter.")
        return 1
    return 1 if run_suite(rag, cases) else 0


if __name__ == "__main__":
    sys.exit(main())