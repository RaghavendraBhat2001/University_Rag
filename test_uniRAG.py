# tests/test_uniRAG.py
"""
Automated tests for uniRAG. They need NO Ollama and NO scanned PDFs, so they run in seconds.

Run from the project root (next to app.py):

    python -m unittest discover -s tests -v

What is covered: topic matching, year detection, question extraction, the exam-history text,
marks filtering, the question-bank protection during rebuilds, login lockout, upload-name cleaning,
and the eval-suite rules. (Answer quality is covered separately by eval_questions.py, which needs Ollama.)
"""
import re
import shutil
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# works whether this file sits in the project root or in tests/
ROOT = _HERE if (_HERE / "app.py").exists() else _HERE.parent
sys.path.insert(0, str(ROOT))


def _source(name):
    """Source text of a project file (root or src/)."""
    for p in (ROOT / name, ROOT / "src" / name):
        if p.exists():
            return p.read_text(encoding="utf-8")
    raise FileNotFoundError(name)


# ---------------------------------------------------------------- topic matching
class TopicMatching(unittest.TestCase):
    def setUp(self):
        from src.question_matcher import _topic_in_text
        self.f = _topic_in_text

    def test_short_topic_needs_whole_word(self):
        self.assertTrue(self.f("os", "explain os scheduling"))
        self.assertFalse(self.f("os", "which is the most asked question"))
        self.assertFalse(self.f("ai", "explain hashing"))

    def test_plural_and_suffix(self):
        self.assertTrue(self.f("stack", "stacks are lifo"))
        self.assertTrue(self.f("hash", "explain hashing"))

    def test_empty_topic_never_matches(self):
        self.assertFalse(self.f("", "anything"))


# ---------------------------------------------------------------- year detection
class YearDetection(unittest.TestCase):
    def setUp(self):
        from src.document_loader import extract_year_from_page_text
        self.f = extract_year_from_page_text

    def test_header_with_months(self):
        self.assertEqual(self.f("MCA Examination, June/July 2024  Max. Marks: 70"), "2024")

    def test_header_with_dotted_months(self):
        self.assertEqual(self.f("MCA Examination, Oct./Nov. 2020"), "2020")

    def test_bare_month_year(self):
        # header word not readable by OCR: month/month + year alone is still enough
        self.assertEqual(self.f("May/June 2025"), "2025")

    def test_no_header_means_no_year(self):
        self.assertIsNone(self.f("Answer any five questions. Each carries 6 marks."))

    def test_scheme_boilerplate_is_not_a_year(self):
        self.assertIsNone(self.f("(2021 - 2022 and Onwards)"))


# ---------------------------------------------------------------- question extraction
class QuestionExtraction(unittest.TestCase):
    PAGE = (
        "Answer any five questions. Each question carries six marks.\n"
        "1. Define a stack and explain the push and pop operations.\n"
        "2. What is a queue? Explain circular queue with an example.\n"
        "3. Explain hashing and collision resolution techniques.\n"
    )

    def setUp(self):
        from src.document_loader import extract_questions_from_text
        self.f = extract_questions_from_text

    def test_rejects_joined_string(self):
        with self.assertRaises(TypeError):
            self.f(self.PAGE)

    def test_extracts_questions_with_section_marks(self):
        qs = self.f([self.PAGE])
        texts = " ".join(q["text"].lower() for q in qs)
        self.assertGreaterEqual(len(qs), 3)
        self.assertIn("stack", texts)
        self.assertIn("circular queue", texts)
        self.assertTrue(all(q["marks"] == 6 for q in qs), [q["marks"] for q in qs])


# ---------------------------------------------------------------- exam intelligence text
class ExamIntelligence(unittest.TestCase):
    def setUp(self):
        from src.question_index import QuestionRecord
        from src.question_matcher import MatchResultSet, QuestionMatch
        from src.exam_intelligence import ExamIntelligence
        self.QR, self.QM, self.MRS, self.EI = QuestionRecord, QuestionMatch, MatchResultSet, ExamIntelligence

    def _rec(self, q, year, marks, subj="DS"):
        return self.QR(subject=subj, question=q, year=year, marks=marks, page=0, source=f"({subj}) x.pdf")

    def test_marks_filter_and_subject_scope(self):
        bank = [self._rec("Define stack", "2023", 5), self._rec("Explain queue", "2022", 10),
                self._rec("What is a process", "2022", 5, "OS")]
        got = self.EI.get_questions_by_marks(bank, 5, subject="DS")
        self.assertEqual([r.question for r in got], ["Define stack"])
        self.assertEqual(len(self.EI.get_questions_by_marks(bank, 5)), 2)  # no subject = all

    def test_history_when_nothing_found(self):
        mr = self.MRS(canonical_topic="Fenwick", total_matching_questions=0, distinct_papers=0,
                      distinct_years=[], matches=[])
        text = self.EI.format_exam_history_section(mr)
        self.assertIn("EXAM HISTORY", text)
        self.assertIn("No questions matching", text)

    def test_history_lists_the_year(self):
        r = self._rec("Define stack. Explain push and pop.", "2023", 5)
        m = self.QM(record=r, match_type="Exact / Near-Exact", score=1.0)
        mr = self.MRS(canonical_topic="Stack", total_matching_questions=1, distinct_papers=1,
                      distinct_years=["2023"], matches=[m], by_year={"2023": [m]},
                      marks_breakdown={5: 1}, subject_scoped=True, total_sittings=6, coverage_subject="DS")
        text = self.EI.format_exam_history_section(mr)
        self.assertIn("2023", text)
        self.assertIn("Define stack", text)


# ---------------------------------------------------------------- "most repeated" ranking
class RankingOnlyRepeats(unittest.TestCase):
    def setUp(self):
        from src.question_index import QuestionRecord
        from src.exam_intelligence import ExamIntelligence
        self.QR, self.EI = QuestionRecord, ExamIntelligence

    def _rec(self, q, year):
        return self.QR(subject="DS", question=q, year=year, marks=6, page=0, source="(DS) x.pdf")

    def test_once_only_questions_are_not_called_repeated(self):
        bank = [
            self._rec("Explain the push and pop operations of a stack with an example", "2022"),
            self._rec("Explain the push and pop operations of a stack with an example", "2024"),
            self._rec("Describe the working of a binary search tree deletion algorithm", "2023"),
        ]
        got = self.EI.get_most_repeated_topics_from_bank(bank, subject="DS")
        self.assertEqual(len(got), 1)
        self.assertIn("push and pop", got[0]["representative_question"])
        self.assertEqual(got[0]["years"], ["2024", "2022"])

    def test_same_year_twice_is_not_a_repeat(self):
        bank = [self._rec("Explain the push and pop operations of a stack with an example", "2022"),
                self._rec("Explain the push and pop operations of a stack with an example", "2022")]
        self.assertEqual(self.EI.get_most_repeated_topics_from_bank(bank, subject="DS"), [])

    def test_old_behaviour_available(self):
        bank = [self._rec("Describe the working of a binary search tree deletion algorithm", "2023")]
        self.assertEqual(len(self.EI.get_most_repeated_topics_from_bank(bank, subject="DS", min_sittings=1)), 1)


# ---------------------------------------------------------------- rebuild protection
class RebuildProtection(unittest.TestCase):
    """The admin 'Rebuild everything' button must never destroy the question bank."""

    def setUp(self):
        src = _source("rag_pipeline.py")
        m = re.search(r"from pathlib import Path as _SnapPath.*?(?=\nclass RAGPipeline:)", src, re.S)
        self.assertIsNotNone(m, "rebuild-protection helpers not found - run patch_hardening.py")
        self.ns = {"time": time, "shutil": shutil, "re": re}
        exec(m.group(0), self.ns)
        self.tmp = Path(tempfile.mkdtemp())
        self.fd = self.tmp / "faiss_store"
        (self.fd / "_question_bank_faiss").mkdir(parents=True)
        (self.fd / "_question_bank.json").write_text('[{"question": "verified"}]')
        (self.fd / "_question_bank_faiss" / "index.faiss").write_text("VEC")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_snapshot_then_restore_after_wipe(self):
        snap = self.ns["_snapshot_question_data"](self.fd)
        self.assertTrue((snap / "_question_bank.json").exists())
        shutil.rmtree(self.fd / "_question_bank_faiss")
        (self.fd / "_question_bank.json").write_text('[{"question": "ocr junk"}]')
        self.ns["_restore_question_data"](snap, self.fd)
        self.assertIn("verified", (self.fd / "_question_bank.json").read_text())
        self.assertEqual((self.fd / "_question_bank_faiss" / "index.faiss").read_text(), "VEC")

    def test_no_bank_means_no_snapshot(self):
        self.assertIsNone(self.ns["_snapshot_question_data"](self.tmp / "nothing"))

    def test_switch_exists_in_config(self):
        import config
        self.assertTrue(getattr(config, "PRESERVE_QUESTION_BANK_ON_REBUILD", None))


# ---------------------------------------------------------------- login lockout
class LoginLockout(unittest.TestCase):
    def setUp(self):
        app = _source("app.py")
        m = re.search(r"_MAX_FAILS = \d+.*?def _login_reset\(key\):\n    _login_attempts\(\)\.pop\(key, None\)\n", app, re.S)
        self.assertIsNotNone(m, "login lockout not found in app.py - run patch_hardening.py")
        store = {}
        st = types.SimpleNamespace(cache_resource=lambda f: (lambda: store.setdefault("v", f())))
        self.ns = {"st": st, "time": time}
        exec(m.group(0), self.ns)

    def test_fifth_wrong_try_locks_only_that_account(self):
        for _ in range(4):
            self.ns["_login_failed"]("a@x.com")
        self.assertEqual(self.ns["_login_locked"]("a@x.com"), 0)
        self.ns["_login_failed"]("a@x.com")
        self.assertGreater(self.ns["_login_locked"]("a@x.com"), 0)
        self.assertEqual(self.ns["_login_locked"]("b@x.com"), 0)

    def test_success_clears_the_counter(self):
        for _ in range(5):
            self.ns["_login_failed"]("a@x.com")
        self.ns["_login_reset"]("a@x.com")
        self.assertEqual(self.ns["_login_locked"]("a@x.com"), 0)

    def test_lock_expires(self):
        for _ in range(5):
            self.ns["_login_failed"]("a@x.com")
        self.ns["_LOCK_SECONDS"] = -1
        self.assertEqual(self.ns["_login_locked"]("a@x.com"), 0)


# ---------------------------------------------------------------- upload name cleaning
class UploadNames(unittest.TestCase):
    def setUp(self):
        app = _source("app.py")
        m = re.search(r"def _clean_name\(s\):.*?return s\.strip\(\)\.strip\(\"\.\"\)\n", app, re.S)
        self.assertIsNotNone(m, "_clean_name not found in app.py - run patch_hardening.py")
        import textwrap
        ns = {"re": re}
        exec(textwrap.dedent(m.group(0)), ns)
        self.clean = ns["_clean_name"]

    def test_path_tricks_are_removed(self):
        self.assertEqual(self.clean("../../etc"), "etc")
        self.assertEqual(self.clean("a\\b/c"), "abc")
        self.assertEqual(self.clean(".."), "")

    def test_normal_names_unchanged(self):
        self.assertEqual(self.clean("Data Structures"), "Data Structures")
        self.assertEqual(self.clean("DS"), "DS")


# ---------------------------------------------------------------- eval-suite rules
class EvalRules(unittest.TestCase):
    def setUp(self):
        import eval_questions as E
        self.E = E

    def test_quick_sort_rule_is_sentence_based(self):
        good = "Best case O(n log n). However, in the worst case it degrades to O(n^2)."
        bad = "In the worst case Quick Sort takes O(n log n) time."
        self.assertTrue(self.E.worst_case_nlogn(good, {})[0])
        self.assertFalse(self.E.worst_case_nlogn(bad, {})[0])

    def test_intent_rule(self):
        rule = self.E.intent_is("exact_questions")
        yes = {"payload": {"debug_info": {"intents": ["exact_questions"]}}}
        no = {"payload": {"debug_info": {"intents": ["definition"]}}}
        self.assertTrue(rule("", yes)[0])
        self.assertFalse(rule("", no)[0])


# ---------------------------------------------------------------- programme questions + diagrams
class ProgrammeQuestions(unittest.TestCase):
    def setUp(self):
        from src import curriculum
        self.c = curriculum
        self.tmp = Path(tempfile.mkdtemp())
        notes, papers = self.tmp / "notes", self.tmp / "papers"
        for d in ("SEM1/(DS) Data Structures", "SEM1/(DSL) Data Structures Lab",
                  "SEM2/(OOP Lab) Object Oriented Programming Lab", "SEM3/(DBMS) Database Management Systems"):
            (notes / d).mkdir(parents=True)
            (notes / d / "x.pdf").write_bytes(b"")
        (papers / "SEM3").mkdir(parents=True)
        (papers / "SEM3" / "(OS) Operating Systems.pdf").write_bytes(b"")
        self.notes, self.papers = notes, papers

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_semester_wording(self):
        f = self.c.find_semester
        self.assertEqual(f("subjects in 1st sem"), 1)
        self.assertEqual(f("subjects in mca 1s sem"), 1)
        self.assertEqual(f("which lab in second semester"), 2)
        self.assertEqual(f("subjects in semester 3"), 3)
        self.assertEqual(f("iii sem subjects"), 3)
        self.assertIsNone(f("what are the subjects"))

    def test_only_programme_questions_are_taken(self):
        p = self.c.parse_curriculum_query
        self.assertIsNotNone(p("how many subjects in 1st sem mca"))
        self.assertIsNotNone(p("which lab in 2nd sem"))
        self.assertIsNone(p("how many times has stack appeared in 1st sem"))   # exam statistics
        self.assertIsNone(p("write a program for linked list lab"))            # lab program
        self.assertIsNone(p("explain hashing"))

    def test_count_and_labs_come_from_the_folders(self):
        cur = self.c.scan_curriculum(self.notes, self.papers)
        self.assertEqual(sorted(cur), [1, 2, 3])
        self.assertEqual(len(cur[1]), 2)
        text = self.c.build_answer({"sem": 1, "labs": False, "count": True}, cur)
        self.assertIn("2 subjects", text)
        labs = self.c.build_answer({"sem": 2, "labs": True, "count": False}, cur)
        self.assertIn("OOP LAB", labs)
        self.assertIn("OS", self.c.build_answer({"sem": 3, "labs": False, "count": False}, cur))

    def test_unknown_semester_is_reported_not_invented(self):
        cur = self.c.scan_curriculum(self.notes, self.papers)
        text = self.c.build_answer({"sem": 6, "labs": False, "count": True}, cur)
        self.assertIn("couldn't find", text)
        self.assertIn("1, 2, 3", text)


class MissingDiagram(unittest.TestCase):
    def setUp(self):
        from src.exam_intelligence import refers_to_missing_diagram
        self.f = refers_to_missing_diagram

    def test_flags_questions_that_need_a_figure(self):
        self.assertTrue(self.f("Find the Hamiltonian path for the following."))
        self.assertTrue(self.f("Consider the following game tree. Find the best move."))
        self.assertTrue(self.f("Solve all pairs shortest path for the below graph."))

    def test_does_not_flag_ordinary_questions(self):
        self.assertFalse(self.f("Define stack. Explain push and pop."))
        self.assertFalse(self.f("Explain the following tree traversal algorithms."))
        self.assertFalse(self.f("Consider the following sentences. a) John likes food."))
        self.assertFalse(self.f("Draw a semantic network representing the following knowledge."))


if __name__ == "__main__":
    unittest.main(verbosity=2)