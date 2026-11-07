import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Clean Academic Palette
    BG_WHITE = RGBColor(255, 255, 255)
    BG_CARD = RGBColor(248, 250, 252)       # Slate 50
    BORDER_CARD = RGBColor(203, 213, 225)   # Slate 300
    TEXT_NAVY = RGBColor(15, 23, 42)        # Slate 900
    TEXT_BODY = RGBColor(51, 65, 85)        # Slate 700
    TEXT_MUTED = RGBColor(100, 116, 139)    # Slate 500
    ACCENT_BLUE = RGBColor(37, 99, 235)     # Blue 600
    ACCENT_DARK_BLUE = RGBColor(30, 58, 138)# Blue 900
    ACCENT_GREEN = RGBColor(16, 149, 102)   # Green
    ACCENT_AMBER = RGBColor(217, 119, 6)    # Amber 600

    def add_slide_base(title_text=""):
        slide = prs.slides.add_slide(blank_layout)
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_WHITE
        bg.line.fill.background()

        if title_text:
            tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.7))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = title_text
            p.font.size = Pt(26)
            p.font.bold = True
            p.font.color.rgb = ACCENT_DARK_BLUE
        return slide

    def add_card_box(slide, left, top, width, height, title="", items=None, bg_color=BG_CARD, border_color=BORDER_CARD, title_color=ACCENT_DARK_BLUE):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)

        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.18), width - Inches(0.4), height - Inches(0.36))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        first_p = True
        if title:
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(15)
            p.font.bold = True
            p.font.color.rgb = title_color
            p.space_after = Pt(6)
            first_p = False

        if items:
            for item in items:
                if first_p:
                    p = tf.paragraphs[0]
                    first_p = False
                else:
                    p = tf.add_paragraph()
                
                if ":" in item and not item.startswith("http"):
                    h, t = item.split(":", 1)
                    run_h = p.add_run()
                    run_h.text = "• " + h.strip() + ": "
                    run_h.font.bold = True
                    run_h.font.size = Pt(12)
                    run_h.font.color.rgb = TEXT_NAVY
                    run_t = p.add_run()
                    run_t.text = t.strip()
                    run_t.font.size = Pt(12)
                    run_t.font.color.rgb = TEXT_BODY
                else:
                    p.text = "• " + item.strip()
                    p.font.size = Pt(12)
                    p.font.color.rgb = TEXT_BODY
                p.space_after = Pt(5)
        return card

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = add_slide_base()
    # Crest
    if os.path.exists("deck_assets/bu_crest.png"):
        s1.shapes.add_picture("deck_assets/bu_crest.png", Inches(5.95), Inches(0.8), width=Inches(1.4))

    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(2.35), Inches(11.333), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.text = "uniRAG"
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = ACCENT_DARK_BLUE

    p = tf1.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    p.text = "An Offline Study Assistant for University Students"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = TEXT_BODY
    p.space_before = Pt(8)

    p = tf1.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    p.text = "Raghavendra Chandrakantha Bhat"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = TEXT_NAVY
    p.space_before = Pt(24)

    p = tf1.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    p.text = "USN P03NK24S126033 · MCA Final Project"
    p.font.size = Pt(14)
    p.font.color.rgb = TEXT_MUTED
    p.space_before = Pt(4)

    p = tf1.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    p.text = "Guide: Dr. Muralidhara B.L · Department of Computer Science and Applications"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE
    p.space_before = Pt(12)

    p = tf1.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    p.text = "Bangalore University · Academic year 2025–2026"
    p.font.size = Pt(14)
    p.font.color.rgb = TEXT_MUTED
    p.space_before = Pt(4)

    s1.notes_slide.notes_text_frame.text = (
        "Good morning respected examiners and my guide Dr. Muralidhara B.L. "
        "I am Raghavendra Chandrakantha Bhat, presenting uniRAG—an offline, retrieval-augmented study assistant "
        "developed for university students, specifically tested on Bangalore University MCA curriculum. "
        "uniRAG grounds answers strictly in university notes, syllabus, and 1,426 historical exam questions, running 100% locally with zero cloud API dependency."
    )

    # =========================================================================
    # SLIDE 2: THE PROBLEM STUDENTS FACE
    # =========================================================================
    s2 = add_slide_base("The problem students face")
    
    # Top banner callout
    banner = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.3), Inches(11.733), Inches(0.8))
    banner.fill.solid()
    banner.fill.fore_color.rgb = RGBColor(241, 245, 249)
    banner.line.color.rgb = BORDER_CARD
    tb_b = s2.shapes.add_textbox(Inches(1.0), Inches(1.4), Inches(11.333), Inches(0.6))
    tf_b = tb_b.text_frame
    p_b = tf_b.paragraphs[0]
    p_b.alignment = PP_ALIGN.CENTER
    r1 = p_b.add_run()
    r1.text = "4.33 crore "
    r1.font.bold = True
    r1.font.size = Pt(16)
    r1.font.color.rgb = ACCENT_BLUE
    r2 = p_b.add_run()
    r2.text = "students enrolled in Indian higher education in 2021-22 (AISHE), each preparing from notes and past papers"
    r2.font.size = Pt(15)
    r2.font.color.rgb = TEXT_NAVY

    # 3 numbered problem columns
    add_card_box(s2, Inches(0.8), Inches(2.4), Inches(3.7), Inches(4.3),
                 title="1. Scattered material",
                 title_color=RGBColor(220, 38, 38),
                 items=[
                     "Notes, syllabus and scanned question papers sit in separate PDFs.",
                     "Finding one topic means reading dozens of pages across multiple unindexed files.",
                     "High cognitive load and wasted revision time during exam weeks."
                 ])

    add_card_box(s2, Inches(4.8), Inches(2.4), Inches(3.7), Inches(4.3),
                 title="2. Chatbots fall short",
                 title_color=RGBColor(217, 119, 6),
                 items=[
                     "Cloud chatbots need continuous active internet.",
                     "Sends private questions and internal materials to external servers.",
                     "Ignores Bangalore University syllabus; invents plausible-sounding but wrong facts (hallucinations)."
                 ])

    add_card_box(s2, Inches(8.8), Inches(2.4), Inches(3.7), Inches(4.3),
                 title="3. No exam statistics",
                 title_color=RGBColor(79, 70, 229),
                 items=[
                     "Commercial LLMs cannot count past university exam questions.",
                     "Nothing tells a student how often a topic was asked, in which exam years, or for how many marks.",
                     "Students lack objective evidence to prioritize high-yield study topics."
                 ])

    s2.notes_slide.notes_text_frame.text = (
        "Highlight the real-world scale: 4.33 crore students in Indian higher education. "
        "Point out the three distinct failure modes of existing study routines: scattered files, limitations of cloud LLMs, and lack of verified exam frequency data."
    )

    # =========================================================================
    # SLIDE 3: WHAT UNIRAG DOES
    # =========================================================================
    s3 = add_slide_base("What uniRAG does")
    
    features = [
        ("Study Assistant", [
            "Explanation, analogy, exam history, trend and study guidance with verified sources.",
            "Two-part answer: Formal academic concept + intuitive real-world analogy."
        ]),
        ("Question Paper Solver", [
            "Upload an entire question paper PDF.",
            "Every question is automatically extracted, categorized, and answered.",
            "Download all comprehensive answers in one click."
        ]),
        ("Exam Intelligence", [
            "Year-wise trend graph for any syllabus topic across 2020–2026.",
            "Most repeated questions ranking across exam sittings.",
            "High-weightage questions filter (10-mark, 6-mark, 5-mark breakdown)."
        ]),
        ("Resource Library", [
            "Browse the indexed notes and past papers of every subject.",
            "Transparent inspection of all underlying reference material."
        ]),
        ("Roles and Security", [
            "Role-based access: Student and administrator accounts.",
            "PBKDF2-HMAC-SHA256 password hashing with random salt.",
            "Brute-force lockout: 5 wrong attempts triggers 60-second cooldown."
        ]),
        ("100% Offline & Private", [
            "Zero cloud API calls, zero telemetry, zero internet required.",
            "All embeddings, vector searches, and LLM inference run strictly on the laptop."
        ])
    ]

    for idx, (f_title, f_items) in enumerate(features):
        r = idx // 3
        c = idx % 3
        left = Inches(0.8 + c * 4.0)
        top = Inches(1.3 + r * 2.8)
        add_card_box(s3, left, top, Inches(3.7), Inches(2.5), title=f_title, items=f_items, title_color=ACCENT_DARK_BLUE)

    s3.notes_slide.notes_text_frame.text = (
        "uniRAG provides an end-to-end suite with 6 core functional modules: Study Assistant, Question Paper Solver, Exam Intelligence, Resource Library, Security, and complete Offline privacy."
    )

    # =========================================================================
    # SLIDE 4: CORE IDEA: THE MODEL WRITES, CODE COUNTS AND CHECKS
    # =========================================================================
    s4 = add_slide_base("Core idea: the model writes, code counts and checks")

    add_card_box(s4, Inches(0.8), Inches(1.3), Inches(5.6), Inches(4.6),
                 title="Language model (Mistral 7B)",
                 title_color=ACCENT_BLUE,
                 items=[
                     "Understands a messy question: Corrects student typos, slang, and mixed-language phrasing.",
                     "Writes the explanation and analogy: Generates clear textbook-grounded theory and intuitive everyday analogies.",
                     "Writes lab programs: Follows the exact C programming style of the official university lab manual.",
                     "Strict Negative Constraint: NEVER produces a count, year or mark."
                 ])

    add_card_box(s4, Inches(6.8), Inches(1.3), Inches(5.7), Inches(4.6),
                 title="Ordinary Python code",
                 title_color=ACCENT_GREEN,
                 items=[
                     "Counts how often a topic was asked: Deterministically scans 1,426 indexed exam questions.",
                     "Lists years, marks and exact questions: Extracts verbatim citations without rounding or guessing.",
                     "Classifies the trend and ranks repeats: Computes trend trajectory (RISING, STABLE, DECLINING).",
                     "Runs the guards that can decline: Evaluates syllabus bounds and similarity distance thresholds."
                 ])

    # Bottom punchline callout
    punch = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.15), Inches(11.733), Inches(0.75))
    punch.fill.solid()
    punch.fill.fore_color.rgb = RGBColor(238, 242, 255)
    punch.line.color.rgb = RGBColor(199, 210, 254)
    tb_p = s4.shapes.add_textbox(Inches(1.0), Inches(6.25), Inches(11.333), Inches(0.55))
    tf_p = tb_p.text_frame
    p_p = tf_p.paragraphs[0]
    p_p.alignment = PP_ALIGN.CENTER
    p_p.text = "No number the student sees comes from the language model."
    p_p.font.size = Pt(17)
    p_p.font.bold = True
    p_p.font.color.rgb = ACCENT_DARK_BLUE

    s4.notes_slide.notes_text_frame.text = (
        "THIS IS THE ARCHITECTURAL HIGHLIGHT of your defense. Tell the examiners: "
        "'Language models are probabilistic token predictors—they should never be asked to do deterministic arithmetic or historical counting. "
        "In uniRAG, the LLM only writes prose explanations; deterministic Python code counts, tracks years, and checks guards.'"
    )

    # =========================================================================
    # SLIDE 5: SYSTEM ARCHITECTURE
    # =========================================================================
    s5 = add_slide_base("System architecture")
    
    if os.path.exists("deck_assets/arch_diagram.png"):
        s5.shapes.add_picture("deck_assets/arch_diagram.png", Inches(1.8), Inches(1.2), width=Inches(9.7))

    # 4 Layers breakdown
    layers = [
        ("Presentation", "Streamlit: login, tabs, charts"),
        ("Application", "Python modules: query understanding, retrieval, matching, statistics"),
        ("Local AI", "Ollama: Mistral 7B and nomic-embed-text"),
        ("Data", "PDFs, per-subject FAISS indexes, JSON question bank")
    ]
    for idx, (l_name, l_desc) in enumerate(layers):
        left = Inches(0.8 + idx * 2.95)
        add_card_box(s5, left, Inches(5.35), Inches(2.8), Inches(1.0), title=l_name, items=[l_desc], title_color=ACCENT_DARK_BLUE)

    foot5 = s5.shapes.add_textbox(Inches(0.8), Inches(6.5), Inches(11.733), Inches(0.4))
    tf_f5 = foot5.text_frame
    p_f5 = tf_f5.paragraphs[0]
    p_f5.alignment = PP_ALIGN.CENTER
    p_f5.text = "Nothing communicates with the internet at run time."
    p_f5.font.size = Pt(14)
    p_f5.font.bold = True
    p_f5.font.color.rgb = ACCENT_GREEN

    s5.notes_slide.notes_text_frame.text = (
        "Explain the 4-tier local architecture: Streamlit at the top, Python application business logic, Ollama hosting Mistral & nomic-embed-text, and local FAISS/JSON storage on disk. Stress that network access is strictly zero."
    )

    # =========================================================================
    # SLIDE 6: HOW ONE ANSWER IS PRODUCED
    # =========================================================================
    s6 = add_slide_base("How one answer is produced")
    
    if os.path.exists("deck_assets/pipeline_diagram.png"):
        s6.shapes.add_picture("deck_assets/pipeline_diagram.png", Inches(0.8), Inches(1.2), width=Inches(11.733))

    add_card_box(s6, Inches(0.8), Inches(2.8), Inches(11.733), Inches(3.3),
                 title="Guards that run before the model writes anything",
                 title_color=ACCENT_DARK_BLUE,
                 items=[
                     "Relevance cut-off ≤ 1.15: Drops retrieved notes chunks if semantic distance exceeds calibrated threshold.",
                     "Topic-in-context check: Confirms the requested concept actually appears inside retrieved syllabus paragraphs.",
                     "Out-of-syllabus gate: Verifies concept presence against syllabus and question bank before calling the LLM.",
                     "Deterministic counting: Direct database match across 1,426 verified exam questions.",
                     "Follow-up check: Disallows unprompted topic carryover between unrelated student questions.",
                     "Figure-reference stripping: Cleans textbook artifacts like 'as shown in Fig 3.2' that confuse students.",
                     "Empty-input guard: Instantly returns polite error on whitespace or invalid prompts."
                 ])

    foot6 = s6.shapes.add_textbox(Inches(0.8), Inches(6.4), Inches(11.733), Inches(0.4))
    tf_f6 = foot6.text_frame
    p_f6 = tf_f6.paragraphs[0]
    p_f6.alignment = PP_ALIGN.CENTER
    p_f6.text = "A declined question never reaches the model, so it is answered instantly."
    p_f6.font.size = Pt(15)
    p_f6.font.bold = True
    p_f6.font.color.rgb = ACCENT_BLUE

    s6.notes_slide.notes_text_frame.text = (
        "Explain the 5-step pipeline: Query Understanding -> Retrieval -> Pre-generation Guards -> Synthesis -> Structured Display. "
        "Highlight that rejected questions take 0.00 seconds because they never invoke the heavy LLM."
    )

    # =========================================================================
    # SLIDE 7: TECHNOLOGY STACK
    # =========================================================================
    s7 = add_slide_base("Technology stack")

    tech_items = [
        ("Python 3.12", "Whole application"),
        ("Streamlit", "Web interface"),
        ("Ollama", "Hosts models locally"),
        ("Mistral 7B", "Parsing, explanations, lab code"),
        ("nomic-embed-text", "768-dimension embeddings"),
        ("FAISS", "Per-subject vector search"),
        ("Tesseract OCR", "Scanned question papers"),
        ("LangChain · PyMuPDF", "Splitting, PDF read and write")
    ]

    for idx, (t_name, t_role) in enumerate(tech_items):
        r = idx // 4
        c = idx % 4
        left = Inches(0.8 + c * 3.0)
        top = Inches(1.5 + r * 2.3)
        add_card_box(s7, left, top, Inches(2.75), Inches(1.9), title=t_name, items=[t_role], title_color=ACCENT_DARK_BLUE)

    foot7 = s7.shapes.add_textbox(Inches(0.8), Inches(6.3), Inches(11.733), Inches(0.5))
    tf_f7 = foot7.text_frame
    p_f7 = tf_f7.paragraphs[0]
    p_f7.alignment = PP_ALIGN.CENTER
    p_f7.text = "Runs on an Intel Core i5-13420H laptop with 16 GB RAM and no GPU. Free, open-source tools only."
    p_f7.font.size = Pt(15)
    p_f7.font.bold = True
    p_f7.font.color.rgb = ACCENT_GREEN

    s7.notes_slide.notes_text_frame.text = (
        "Present your hardware specs clearly: Runs on Intel Core i5-13420H, 16 GB RAM, zero GPU. All tools used are 100% free and open source."
    )

    # =========================================================================
    # SLIDE 8: FROM SCANNED PAPERS TO A QUESTION BANK
    # =========================================================================
    s8 = add_slide_base("From scanned papers to a question bank")

    kpis = [
        ("1,426", "questions"),
        ("18", "subjects"),
        ("7", "exam years (2020–2026)"),
        ("125", "repeated questions")
    ]
    for idx, (num, lbl) in enumerate(kpis):
        left = Inches(0.8 + idx * 3.0)
        box = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.3), Inches(2.75), Inches(1.3))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(239, 246, 255)
        box.line.color.rgb = RGBColor(191, 219, 254)
        tb_k = s8.shapes.add_textbox(left, Inches(1.35), Inches(2.75), Inches(1.2))
        tf_k = tb_k.text_frame
        p_n = tf_k.paragraphs[0]
        p_n.alignment = PP_ALIGN.CENTER
        p_n.text = num
        p_n.font.size = Pt(30)
        p_n.font.bold = True
        p_n.font.color.rgb = ACCENT_BLUE
        p_l = tf_k.add_paragraph()
        p_l.alignment = PP_ALIGN.CENTER
        p_l.text = lbl
        p_l.font.size = Pt(13)
        p_l.font.bold = True
        p_l.font.color.rgb = TEXT_NAVY

    add_card_box(s8, Inches(0.8), Inches(2.85), Inches(11.733), Inches(3.2),
                 title="Four-level hybrid question matcher",
                 title_color=ACCENT_DARK_BLUE,
                 items=[
                     "1  Exact: Same words in the question (substring exact match and high-ratio fuzzy match).",
                     "2  Subtopic: Topic and related subtopics (e.g. Circular Queue vs Linear Queue).",
                     "3  Semantic: Embedding distance ≤ 0.75 via nomic-embed-text dense vectors.",
                     "4  Filters: Marks and year constraints (e.g. 10-mark questions only)."
                 ])

    foot8 = s8.shapes.add_textbox(Inches(0.8), Inches(6.3), Inches(11.733), Inches(0.5))
    tf_f8 = foot8.text_frame
    p_f8 = tf_f8.paragraphs[0]
    p_f8.alignment = PP_ALIGN.CENTER
    p_f8.text = "OCR reads scanned pages; the exam year is taken from the printed header and carried over continuation pages."
    p_f8.font.size = Pt(14)
    p_f8.font.bold = True
    p_f8.font.color.rgb = TEXT_BODY

    s8.notes_slide.notes_text_frame.text = (
        "Showcase your dataset size: 1,426 questions, 18 subjects, 7 exam years, 125 repeated questions. "
        "Explain the OCR header logic: university exam PDFs merge multiple years into one PDF. Tesseract parses the printed header ('Examination, June 2024') and carries it across continuation pages."
    )

    # =========================================================================
    # SLIDE 9: CHOOSING THE MATCHING CUT-OFF BY MEASUREMENT
    # =========================================================================
    s9 = add_slide_base("Choosing the matching cut-off by measurement")

    if os.path.exists("deck_assets/f1_cutoff.png"):
        s9.shapes.add_picture("deck_assets/f1_cutoff.png", Inches(3.5), Inches(1.15), width=Inches(6.3))

    sub_t = s9.shapes.add_textbox(Inches(0.8), Inches(5.1), Inches(11.733), Inches(0.4))
    tf_sub = sub_t.text_frame
    p_s = tf_sub.paragraphs[0]
    p_s.alignment = PP_ALIGN.CENTER
    p_s.text = "Labelled set of 61 queries: 46 with a correct stored question, 15 that should match nothing."
    p_s.font.size = Pt(14)
    p_s.font.bold = True
    p_s.font.color.rgb = TEXT_BODY

    cut_kpis = [
        ("F1 score", "0.435 → 0.757"),
        ("Cut-off", "1.15 → 0.75"),
        ("Wrong questions shown", "665 → 61"),
        ("Nothing-queries answered", "15 of 15 → 1 of 15")
    ]
    for idx, (title_k, val_k) in enumerate(cut_kpis):
        left = Inches(0.8 + idx * 3.0)
        box = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(5.6), Inches(2.75), Inches(1.15))
        box.fill.solid()
        box.fill.fore_color.rgb = BG_CARD
        box.line.color.rgb = BORDER_CARD
        tb_ck = s9.shapes.add_textbox(left, Inches(5.65), Inches(2.75), Inches(1.0))
        tf_ck = tb_ck.text_frame
        p1 = tf_ck.paragraphs[0]
        p1.alignment = PP_ALIGN.CENTER
        p1.text = title_k
        p1.font.size = Pt(12)
        p1.font.color.rgb = TEXT_MUTED
        p2 = tf_ck.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        p2.text = val_k
        p2.font.size = Pt(16)
        p2.font.bold = True
        p2.font.color.rgb = ACCENT_BLUE

    s9.notes_slide.notes_text_frame.text = (
        "Explain calibration: You calibrated distance cutoff against a ground-truth dataset of 61 test queries. "
        "At the default cut-off (1.15), F1 was only 0.435 and 665 wrong questions leaked through. "
        "Tightening the threshold to 0.75 increased F1 to 0.757 and slashed false positive nothing-queries from 15 down to 1."
    )

    # =========================================================================
    # SLIDE 10: DEMO 1: STUDY ASSISTANT
    # =========================================================================
    s10 = add_slide_base("Demo 1: Study Assistant")

    if os.path.exists("deck_assets/demo1_concept_stack.png"):
        s10.shapes.add_picture("deck_assets/demo1_concept_stack.png", Inches(0.8), Inches(1.2), width=Inches(5.6))

    if os.path.exists("deck_assets/demo1_exam_history.png"):
        s10.shapes.add_picture("deck_assets/demo1_exam_history.png", Inches(6.8), Inches(1.2), width=Inches(5.7))

    demo1_box = s10.shapes.add_textbox(Inches(0.8), Inches(5.8), Inches(11.733), Inches(1.2))
    tf_d1 = demo1_box.text_frame
    tf_d1.word_wrap = True
    
    p = tf_d1.paragraphs[0]
    p.text = "• Concept and analogy from the notes, with sources"
    p.font.size = Pt(14)
    p.font.color.rgb = TEXT_NAVY

    p = tf_d1.add_paragraph()
    p.text = "• Exam history computed by code: 5 sittings, years, marks, priority tier"
    p.font.size = Pt(14)
    p.font.color.rgb = TEXT_NAVY

    p = tf_d1.add_paragraph()
    p.text = "• Try: “what is stack?”, then “what is a Fenwick tree” (declined instantly)."
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE

    s10.notes_slide.notes_text_frame.text = (
        "Walk examiners through the UI: 1) Query: 'what is stack?' produces academic definition + real-world dinner plate analogy. "
        "2) Below it, Python outputs exact history: appeared in 5 exam sittings (2025, 2024, 2023, 2022, 2020) with marks. "
        "3) Show instant refusal with 'what is a Fenwick tree'."
    )

    # =========================================================================
    # SLIDE 11: DEMO 2: EXAM INTELLIGENCE
    # =========================================================================
    s11 = add_slide_base("Demo 2: Exam Intelligence")

    if os.path.exists("deck_assets/demo2_topic_trend.png"):
        s11.shapes.add_picture("deck_assets/demo2_topic_trend.png", Inches(0.8), Inches(1.2), width=Inches(3.7))

    if os.path.exists("deck_assets/demo2_repeated_questions.png"):
        s11.shapes.add_picture("deck_assets/demo2_repeated_questions.png", Inches(4.8), Inches(1.2), width=Inches(3.7))

    if os.path.exists("deck_assets/demo2_high_weightage.png"):
        s11.shapes.add_picture("deck_assets/demo2_high_weightage.png", Inches(8.8), Inches(1.2), width=Inches(3.7))

    d2_box = s11.shapes.add_textbox(Inches(0.8), Inches(5.6), Inches(11.733), Inches(1.3))
    tf_d2 = d2_box.text_frame
    tf_d2.word_wrap = True

    p = tf_d2.paragraphs[0]
    p.text = "• Topic trend: Queue across 2020–2025 (stable)"
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_NAVY

    p = tf_d2.add_paragraph()
    p.text = "• Most repeated questions with a solutions PDF bundle"
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_NAVY

    p = tf_d2.add_paragraph()
    p.text = "• High-weightage questions by marks (10-mark, 6-mark, 5-mark distribution)"
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_NAVY

    p = tf_d2.add_paragraph()
    p.text = "All three views use only the selected subject. A zero-filled year means no matching question that year, not a missing chart. 125 repeated questions: 67 curated answers, 58 generated by the local model."
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ACCENT_DARK_BLUE

    s11.notes_slide.notes_text_frame.text = (
        "Explain the three analytical views: Year-by-Year topic trend, Most Repeated Questions bundle, and High-Weightage questions breakdown. Mention that 67 model answers are human-curated and 58 were locally synthesized."
    )

    # =========================================================================
    # SLIDE 12: PERFORMANCE ON A CPU-ONLY LAPTOP
    # =========================================================================
    s12 = add_slide_base("Performance on a CPU-only laptop")

    perf_cards = [
        ("25 s", "exam-fact answers\n(counts by code)"),
        ("1–3.5 min", "explanations and lab programs\n(Mistral 7B inference)"),
        ("< 1 s", "saved answers\nfrom the cache")
    ]

    for idx, (p_num, p_lbl) in enumerate(perf_cards):
        left = Inches(0.8 + idx * 4.0)
        box = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(2.2), Inches(3.7), Inches(3.2))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(248, 250, 252)
        box.line.color.rgb = BORDER_CARD
        box.line.width = Pt(1.5)
        
        tb = s12.shapes.add_textbox(left + Inches(0.2), Inches(2.6), Inches(3.3), Inches(2.4))
        tf = tb.text_frame
        p1 = tf.paragraphs[0]
        p1.alignment = PP_ALIGN.CENTER
        p1.text = p_num
        p1.font.size = Pt(44)
        p1.font.bold = True
        p1.font.color.rgb = ACCENT_BLUE

        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        p2.text = p_lbl
        p2.font.size = Pt(16)
        p2.font.color.rgb = TEXT_NAVY
        p2.space_before = Pt(12)

    s12.notes_slide.notes_text_frame.text = (
        "Deliver the hardware latency explanation: Memory bandwidth is ~30 GB/s on dual-channel RAM. "
        "Quantized weights take 4.1 GB. Therefore CPU generation speed is ~7 tokens/sec. "
        "Prose answers take 1-3.5 mins, code counting takes ~25s, and cached answers return in under 1 second."
    )

    # =========================================================================
    # SLIDE 13: TESTING
    # =========================================================================
    s13 = add_slide_base("Testing")

    add_card_box(s13, Inches(0.8), Inches(1.5), Inches(3.7), Inches(4.5),
                 title="32 automated unit tests",
                 title_color=ACCENT_BLUE,
                 items=[
                     "Topic matching validation.",
                     "Exam year detection from OCR headers.",
                     "Question text extraction and cleanup.",
                     "Login brute-force lockout rules.",
                     "Upload file sanitization and name cleaning."
                 ])

    add_card_box(s13, Inches(4.8), Inches(1.5), Inches(3.7), Inches(4.5),
                 title="29 regression cases (44 checks)",
                 title_color=ACCENT_GREEN,
                 items=[
                     "No topic carry-over across follow-ups.",
                     "Refusal of out-of-syllabus topics.",
                     "Consistent counts and year alignment.",
                     "Lab program language validation (pure C).",
                     "Repeatable and deterministic answers."
                 ])

    add_card_box(s13, Inches(8.8), Inches(1.5), Inches(3.7), Inches(4.5),
                 title="27 manual test cases",
                 title_color=ACCENT_AMBER,
                 items=[
                     "Student self-registration & login.",
                     "Role-based privilege separation.",
                     "Question Paper Solver pipeline.",
                     "Exam Intelligence charts rendering.",
                     "OCR rebuilds & handling of corrupt PDFs."
                 ])

    foot13 = s13.shapes.add_textbox(Inches(0.8), Inches(6.3), Inches(11.733), Inches(0.5))
    tf_f13 = foot13.text_frame
    p_f13 = tf_f13.paragraphs[0]
    p_f13.alignment = PP_ALIGN.CENTER
    p_f13.text = "Defects found in testing, such as a topic carried into an unrelated question, were fixed and the tests repeated."
    p_f13.font.size = Pt(14)
    p_f13.font.bold = True
    p_f13.font.color.rgb = TEXT_BODY

    s13.notes_slide.notes_text_frame.text = (
        "Demonstrate professional engineering rigor: 32 unit tests, 29 regression cases (44 programmatic assertion checks via eval_questions.py), and 27 manual test cases."
    )

    # =========================================================================
    # SLIDE 14: SECURITY, LIMITATIONS AND FUTURE WORK
    # =========================================================================
    s14 = add_slide_base("Security, limitations and future work")

    add_card_box(s14, Inches(0.8), Inches(1.5), Inches(3.7), Inches(4.8),
                 title="Security",
                 title_color=ACCENT_DARK_BLUE,
                 items=[
                     "PBKDF2-HMAC-SHA256: 100,000 iterations with random cryptographic salt.",
                     "Account Lockout: 5 consecutive failed passwords enforces 60-second cooldown.",
                     "Atomic Writes: Safe JSON and index writes with automatic backup rotation.",
                     "Admin Auth: Admin password strictly controlled via environment variables."
                 ])

    add_card_box(s14, Inches(4.8), Inches(1.5), Inches(3.7), Inches(4.8),
                 title="Limitations",
                 title_color=RGBColor(220, 38, 38),
                 items=[
                     "Literal Counting: Very unusual question re-phrasings can occasionally miss exact matching.",
                     "Small Model Slips: 7B models can occasionally slip on minor factual nuances.",
                     "OCR Noise: Low-resolution historical scans may introduce typographical noise.",
                     "No Figure Extraction: Diagrams in scanned questions are not yet extracted as vector graphics."
                 ])

    add_card_box(s14, Inches(8.8), Inches(1.5), Inches(3.7), Inches(4.8),
                 title="Future work",
                 title_color=ACCENT_GREEN,
                 items=[
                     "Sentence-Level Verification: Cross-verifying each generated sentence with notes.",
                     "GPU / Model Speed: Adapting quantized models or lightweight GPUs for sub-10s answers.",
                     "Regional Language Support: Kannada and Hindi prompt & answer translation.",
                     "Campus Deployment: Multi-user student study network over university intranet."
                 ])

    s14.notes_slide.notes_text_frame.text = (
        "Examiners appreciate an honest assessment of limitations: discuss PBKDF2 security, acknowledge OCR noise and 7B model bounds, and conclude with the roadmap for regional languages and campus intranet deployment."
    )

    # =========================================================================
    # SLIDE 15: CONCLUSION
    # =========================================================================
    s15 = add_slide_base("Conclusion")

    add_card_box(s15, Inches(0.8), Inches(1.5), Inches(3.7), Inches(3.8),
                 title="Private and free",
                 title_color=ACCENT_DARK_BLUE,
                 items=[
                     "Runs fully offline on a student laptop.",
                     "Zero subscription fees and zero commercial API bills.",
                     "100% student data privacy."
                 ])

    add_card_box(s15, Inches(4.8), Inches(1.5), Inches(3.7), Inches(3.8),
                 title="Trustworthy numbers",
                 title_color=ACCENT_GREEN,
                 items=[
                     "Counts, years and trends come strictly from Python code.",
                     "1,426 verified Bangalore University MCA questions.",
                     "Zero statistical hallucination."
                 ])

    add_card_box(s15, Inches(8.8), Inches(1.5), Inches(3.7), Inches(3.8),
                 title="Measured, not guessed",
                 title_color=ACCENT_AMBER,
                 items=[
                     "Cut-off calibrated: F1 score improved from 0.435 to 0.757.",
                     "29 regression test cases with 44 checks.",
                     "Systematic verification of all guards."
                 ])

    paper_note = s15.shapes.add_textbox(Inches(0.8), Inches(5.6), Inches(11.733), Inches(0.4))
    tf_pn = paper_note.text_frame
    p_pn = tf_pn.paragraphs[0]
    p_pn.alignment = PP_ALIGN.CENTER
    p_pn.text = "A research paper on uniRAG has also been written in IEEE style."
    p_pn.font.size = Pt(15)
    p_pn.font.bold = True
    p_pn.font.color.rgb = ACCENT_BLUE

    closing = s15.shapes.add_textbox(Inches(0.8), Inches(6.1), Inches(11.733), Inches(0.8))
    tf_cl = closing.text_frame
    p_cl = tf_cl.paragraphs[0]
    p_cl.alignment = PP_ALIGN.CENTER
    p_cl.text = "Thank you. Questions?"
    p_cl.font.size = Pt(28)
    p_cl.font.bold = True
    p_cl.font.color.rgb = ACCENT_DARK_BLUE

    s15.notes_slide.notes_text_frame.text = (
        "Conclude confidently: 'In conclusion, uniRAG demonstrates that a university study assistant can be privacy-first, free, and mathematically trustworthy. "
        "We have also formalized these findings into an IEEE-format research paper. Thank you, I am now ready for your questions.'"
    )

    out_file = "uniRAG_Official_Lab_Exam_Deck.pptx"
    prs.save(out_file)
    print(f"Exact 15-Slide Presentation built and saved to: {os.path.abspath(out_file)}")

if __name__ == "__main__":
    build_presentation()
