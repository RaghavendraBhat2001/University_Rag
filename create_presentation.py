import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color Palette - Professional Tech Slate / Navy & Vibrant Accents
    BG_COLOR = RGBColor(15, 23, 42)        # Slate 900
    CARD_BG = RGBColor(30, 41, 59)         # Slate 800
    CARD_BORDER = RGBColor(51, 65, 85)     # Slate 700
    ACCENT_BLUE = RGBColor(56, 189, 248)   # Sky 400
    ACCENT_INDIGO = RGBColor(129, 140, 248)# Indigo 400
    ACCENT_GREEN = RGBColor(52, 211, 153)  # Emerald 400
    ACCENT_AMBER = RGBColor(251, 191, 36)  # Amber 400
    TEXT_MAIN = RGBColor(248, 250, 252)    # Slate 50
    TEXT_MUTED = RGBColor(148, 163, 184)   # Slate 400
    TEXT_DARK = RGBColor(15, 23, 42)

    blank_layout = prs.slide_layouts[6]

    def add_blank_slide_with_bg():
        slide = prs.slides.add_slide(blank_layout)
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.fill.background()
        return slide

    def add_header(slide, title_text, category="uniRAG LAB EXAM DEFENSE"):
        # Category pill/breadcrumb
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_BLUE

        # Title
        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.65))
        tf_t = t_box.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(24)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_MAIN

    def add_card(slide, left, top, width, height, title="", items=None, border_color=CARD_BORDER, bg_color=CARD_BG, title_color=ACCENT_BLUE):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.5)

        tb = slide.shapes.add_textbox(left + Inches(0.25), top + Inches(0.2), width - Inches(0.5), height - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        first_p = True
        if title:
            p_title = tf.paragraphs[0]
            p_title.text = title
            p_title.font.size = Pt(16)
            p_title.font.bold = True
            p_title.font.color.rgb = title_color
            p_title.space_after = Pt(10)
            first_p = False

        if items:
            for itm in items:
                if first_p:
                    p = tf.paragraphs[0]
                    first_p = False
                else:
                    p = tf.add_paragraph()
                
                # Check if it's a bold header + text
                if ":" in itm and not itm.startswith("http"):
                    header_part, text_part = itm.split(":", 1)
                    run_h = p.add_run()
                    run_h.text = "• " + header_part + ":"
                    run_h.font.bold = True
                    run_h.font.size = Pt(13)
                    run_h.font.color.rgb = TEXT_MAIN

                    run_t = p.add_run()
                    run_t.text = text_part
                    run_t.font.size = Pt(13)
                    run_t.font.color.rgb = TEXT_MUTED
                else:
                    p.text = "• " + itm
                    p.font.size = Pt(13)
                    p.font.color.rgb = TEXT_MAIN
                
                p.space_after = Pt(8)
        return card

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    slide1 = add_blank_slide_with_bg()

    # Decorative top bar
    bar = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.0), Inches(2.5), Inches(0.08))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT_BLUE
    bar.line.fill.background()

    # Main Title box
    t_box = slide1.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(11.5), Inches(2.2))
    tf = t_box.text_frame
    tf.word_wrap = True
    
    p0 = tf.paragraphs[0]
    p0.text = "uniRAG"
    p0.font.size = Pt(48)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT_BLUE

    p1 = tf.add_paragraph()
    p1.text = "Offline Retrieval-Augmented Study Assistant for University Students"
    p1.font.size = Pt(26)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_MAIN
    p1.space_before = Pt(6)

    p2 = tf.add_paragraph()
    p2.text = "100% Local, Zero-Cloud, Syllabus-Grounded Exam Intelligence & Answer Generation"
    p2.font.size = Pt(16)
    p2.font.color.rgb = TEXT_MUTED
    p2.space_before = Pt(8)

    # Details Card - Candidate & Guide Info
    card_info = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.9), Inches(11.733), Inches(2.8))
    card_info.fill.solid()
    card_info.fill.fore_color.rgb = CARD_BG
    card_info.line.color.rgb = CARD_BORDER
    card_info.line.width = Pt(1.5)

    info_tb = slide1.shapes.add_textbox(Inches(1.2), Inches(4.2), Inches(10.9), Inches(2.2))
    itf = info_tb.text_frame
    itf.word_wrap = True

    # 2 columns inside the card represented cleanly
    ip1 = itf.paragraphs[0]
    r = ip1.add_run()
    r.text = "Candidate: "
    r.font.bold = True
    r.font.size = Pt(15)
    r.font.color.rgb = ACCENT_INDIGO
    r2 = ip1.add_run()
    r2.text = "Raghavendra Chandrakantha Bhat   |   Reg No: P03NK24S126033\n"
    r2.font.size = Pt(15)
    r2.font.color.rgb = TEXT_MAIN

    ip2 = itf.add_paragraph()
    r = ip2.add_run()
    r.text = "Programme: "
    r.font.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = ACCENT_INDIGO
    r2 = ip2.add_run()
    r2.text = "Master of Computer Applications (MCA), Department of Computer Science & Applications\n"
    r2.font.size = Pt(14)
    r2.font.color.rgb = TEXT_MAIN

    ip3 = itf.add_paragraph()
    r = ip3.add_run()
    r.text = "Institution: "
    r.font.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = ACCENT_INDIGO
    r2 = ip3.add_run()
    r2.text = "Bangalore University, Jnanabharathi Campus, Bengaluru\n"
    r2.font.size = Pt(14)
    r2.font.color.rgb = TEXT_MAIN

    ip4 = itf.add_paragraph()
    r = ip4.add_run()
    r.text = "Project Guide: "
    r.font.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = ACCENT_GREEN
    r2 = ip4.add_run()
    r2.text = "Dr. Muralidhara B.L., Professor, Bangalore University"
    r2.font.size = Pt(14)
    r2.font.color.rgb = TEXT_MAIN

    # Notes
    slide1.notes_slide.notes_text_frame.text = (
        "Good morning/afternoon respected examiners and guide Dr. Muralidhara B.L. "
        "I am Raghavendra Chandrakantha Bhat, presenting uniRAG—an offline, retrieval-augmented study assistant "
        "designed specifically for university students. uniRAG solves the exam prep dilemma by grounding answers "
        "purely in university notes, syllabus, and 1,426 historical exam questions, running 100% locally with zero cloud dependence."
    )

    # =========================================================================
    # SLIDE 2: PROBLEM STATEMENT & MOTIVATION
    # =========================================================================
    slide2 = add_blank_slide_with_bg()
    add_header(slide2, "The Problem & Why General AI Fails University Exams")

    add_card(slide2, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             title="The Student's Dilemma",
             title_color=RGBColor(248, 113, 113), # Red 400
             items=[
                 "Fragmented Study Material: Syllabus, lecture notes, textbook slides, and past papers scattered across physical photocopies and unstructured PDFs.",
                 "Scanned Question Papers: Historical papers exist as scanned images without searchable text, making revision tedious.",
                 "Subject-Specific Mark Schemes: Students struggle to know how deeply to answer for 2-mark vs 5-mark vs 10-mark questions.",
                 "Prioritization Confusion: No easy way to know which concepts are high-frequency vs rarely tested in actual university exams."
             ])

    add_card(slide2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2),
             title="Why Cloud LLMs (ChatGPT/Gemini) Fall Short",
             title_color=RGBColor(251, 146, 60), # Orange 400
             items=[
                 "Internet & Subscription Barrier: Requires active internet and expensive subscriptions inaccessible during offline campus study.",
                 "Data Privacy & Leaks: Student queries and internal university course materials sent to third-party commercial servers.",
                 "Hallucination of Exam Metrics: LLMs are probabilistic text generators—they CANNOT accurately count how many times a question appeared in past papers.",
                 "Out-of-Syllabus Answers: General models explain topics using foreign syllabi and unverified web sources rather than Bangalore University prescribed notes."
             ])

    slide2.notes_slide.notes_text_frame.text = (
        "Examiners often ask why students don't simply use ChatGPT. "
        "Highlight the two distinct issues: First, practical issues—students need offline access and privacy. "
        "Second, technical flaws of LLMs: LLMs hallucinate statistical numbers (frequency of questions) and lack grounding in BU's exact syllabus."
    )

    # =========================================================================
    # SLIDE 3: CORE PHILOSOPHY & UNIQUE VALUE PROPOSITION
    # =========================================================================
    slide3 = add_blank_slide_with_bg()
    add_header(slide3, "Core Philosophy: 'The LLM Writes, Deterministic Code Counts'")

    add_card(slide3, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.2),
             title="1. Generative LLM Role",
             title_color=ACCENT_INDIGO,
             items=[
                 "Task: Natural language explanation and conceptual synthesis.",
                 "Input: Strictly bounded context from retrieved university notes.",
                 "Two-Part Output Structure: 1) Formal textbook-grounded academic definition, and 2) Intuitive real-world analogy.",
                 "Code Generation: Strict generation of academic C code following the official MCA lab manual approach.",
                 "Rule: The LLM is NEVER asked to compute numbers or remember dates."
             ])

    add_card(slide3, Inches(4.8), Inches(1.6), Inches(3.6), Inches(5.2),
             title="2. Deterministic Code Role",
             title_color=ACCENT_GREEN,
             items=[
                 "Task: Counting, statistics, trend ranking, and consistency.",
                 "Data Source: 1,426 structured questions extracted from verified BU exam papers (2020–2026).",
                 "Hybrid Question Matcher: Substring exact match + subtopic check + semantic vector search.",
                 "Zero Hallucination: Verbatim citation of exam sitting, year, semester, and marks allocation.",
                 "Mathematical Certainty: Exact sitting counts calculated in 0.00 seconds."
             ])

    add_card(slide3, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.2),
             title="3. Zero-Latency Guardrails",
             title_color=ACCENT_AMBER,
             items=[
                 "Syllabus Verification Gate: Checks query against OCR-extracted syllabus before touching the LLM.",
                 "Instant Out-of-Scope Rejection: Off-topic questions (e.g. quantum physics in MCA) rejected in 0.00s.",
                 "Confidence Distance Threshold: Rejects retrieval chunks if cosine distance exceeds 0.72.",
                 "Hardware Safeguard: Prevents wasting 1-2 minutes of CPU inference on invalid queries."
             ])

    # =========================================================================
    # SLIDE 4: SYSTEM ARCHITECTURE
    # =========================================================================
    slide4 = add_blank_slide_with_bg()
    add_header(slide4, "End-to-End System Architecture & Pipeline")

    # 4 Architecture Stages
    add_card(slide4, Inches(0.8), Inches(1.6), Inches(2.7), Inches(5.2),
             title="Stage 1: Ingestion",
             title_color=ACCENT_BLUE,
             items=[
                 "Document Sources: Notes, Syllabus, Past Papers.",
                 "Tesseract OCR: Automatically reads scanned papers with no text layer.",
                 "Header Extraction: Identifies exam year, month, and semester via regex.",
                 "Question Bank Builder: Creates structured JSON of 1,426 questions."
             ])

    add_card(slide4, Inches(3.8), Inches(1.6), Inches(2.7), Inches(5.2),
             title="Stage 2: Indexing",
             title_color=ACCENT_INDIGO,
             items=[
                 "nomic-embed-text: Generates 768-dim dense vector embeddings.",
                 "Isolated FAISS Indexes: Independent vectorstore per subject (prevents topic crowding).",
                 "Question Vectorstore: Dedicated index for semantic past-exam search.",
                 "Disk Footprint: Lightweight indexes in faiss_store/."
             ])

    add_card(slide4, Inches(6.8), Inches(1.6), Inches(2.7), Inches(5.2),
             title="Stage 3: Hybrid Retrieval",
             title_color=ACCENT_GREEN,
             items=[
                 "Query Understanding: Classifies intent (explain, count, code).",
                 "Multi-Level Matcher: Exact keyword -> Subtopic -> Semantic vector search.",
                 "Cross-Subject Fallback: Automatically searches other subjects if 0 matches found.",
                 "Top-K Notes Retrieval: Fetches 3 most relevant paragraphs."
             ])

    add_card(slide4, Inches(9.8), Inches(1.6), Inches(2.7), Inches(5.2),
             title="Stage 4: Generation",
             title_color=ACCENT_AMBER,
             items=[
                 "Mistral 7B (Instruct): 4-bit quantized (Q4_K_M) local model.",
                 "Ollama Engine: Local inference server on CPU.",
                 "Token Capping: Marks-based budget (2M=250, 5M=600, 10M=1100).",
                 "Real-Time Stream: Tokens streamed directly via Streamlit."
             ])

    # =========================================================================
    # SLIDE 5: TECH STACK & HARDWARE SPECIFICATIONS
    # =========================================================================
    slide5 = add_blank_slide_with_bg()
    add_header(slide5, "Technical Stack & Hardware Constraints")

    add_card(slide5, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             title="Open-Source Software Stack",
             title_color=ACCENT_BLUE,
             items=[
                 "Language Model: Mistral 7B Instruct (Q4_K_M quantized GGUF format).",
                 "Embedding Model: nomic-embed-text (8,192 token context window, 768 dimensions).",
                 "Vector Store: FAISS (Facebook AI Similarity Search) - IndexFlatL2 & IndexFlatIP.",
                 "Model Serving: Ollama runtime (cross-platform, zero cloud cost).",
                 "Document Extraction: Tesseract OCR + pdfplumber + PyPDF2.",
                 "Application Interface: Streamlit web framework with real-time SSE streaming.",
                 "Security: Salted PBKDF2 SHA-256 password hashing for student/admin auth."
             ])

    add_card(slide5, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2),
             title="Hardware Profile & Practical Feasibility",
             title_color=ACCENT_GREEN,
             items=[
                 "Deployment Target: Standard Consumer Laptop (No dedicated GPU required).",
                 "System RAM: 16 GB DDR4/DDR5 system memory.",
                 "CPU Execution: Runs on Intel/AMD multi-core CPUs via vectorized AVX2/AVX-512.",
                 "Memory Consumption: Quantized Mistral uses ~4.1 GB RAM (leaving 10+ GB for OS & UI).",
                 "Zero Subscriptions: 100% free open-source software stack with no recurring API bills.",
                 "Complete Data Privacy: Works fully offline without an internet connection."
             ])

    # =========================================================================
    # SLIDE 6: SOLVING THE CPU INFERENCE BOTTLENECK
    # =========================================================================
    slide6 = add_blank_slide_with_bg()
    add_header(slide6, "Deep Dive: Why CPU Takes Time & How uniRAG Solved It")

    add_card(slide6, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             title="The CPU Memory Bandwidth Bottleneck",
             title_color=RGBColor(251, 146, 60),
             items=[
                 "Autoregressive Generation: LLMs generate text token-by-token. A 500-token answer requires 500 sequential forward passes.",
                 "Memory Weight Streaming: In EVERY forward pass for ONE token, all 4.1 GB of weights must be fetched from system RAM into CPU cache.",
                 "Hardware Bandwidth Limit: Dual-channel DDR4/DDR5 RAM bandwidth is ~25 to 40 GB/s (vs GPU VRAM at 1,000 GB/s).",
                 "Theoretical Generation Speed: 30 GB/s / 4.1 GB weights = 7 to 8 tokens per second on CPU.",
                 "Realistic Time: 700 tokens / 7.5 tokens/sec ≈ 90 to 100 seconds (~1.5 min) on CPU."
             ])

    add_card(slide6, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2),
             title="5 Strategic Mitigations Engineered in uniRAG",
             title_color=ACCENT_GREEN,
             items=[
                 "1. Streaming UI (st.write_stream): Tokens appear immediately as generated; student reads without waiting for full generation.",
                 "2. Marks-Based Token Capping: 2 marks = 250 tokens (~25s); 5 marks = 600 tokens (~60s); 10 marks = 1,100 tokens (~110s).",
                 "3. Model Warmth (LLM_KEEP_ALIVE = '30m'): Ollama holds Mistral in RAM, eliminating 20s cold-load disk delays.",
                 "4. In-Memory Answer Cache: Previously answered questions load instantly in < 0.10 seconds.",
                 "5. Deterministic Instant Stats: Exam statistics and frequency load in 0.01 seconds before the LLM generates a single token."
             ])

    slide6.notes_slide.notes_text_frame.text = (
        "This slide directly answers the external examiner's favorite question: "
        "'Why does your model take 1-2 minutes to answer on a laptop?' "
        "Explain the memory bandwidth bottleneck: 30 GB/s divided by 4.1 GB weights gives ~7 tokens/sec. "
        "Then show the 5 mitigations you engineered: streaming, token capping, keep-alive, answer cache, and instant stats."
    )

    # =========================================================================
    # SLIDE 7: HYBRID RETRIEVAL & QUESTION MATCHING
    # =========================================================================
    slide7 = add_blank_slide_with_bg()
    add_header(slide7, "Hybrid Retrieval: Multi-Level Question Matching Engine")

    add_card(slide7, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             title="The 4-Level Cascading Matcher",
             title_color=ACCENT_INDIGO,
             items=[
                 "Level 1: Exact & Substring Match: High-precision match against question text, acronyms, and syllabus keywords.",
                 "Level 2: Subtopic Discrimination: Distinguishes fine-grained topics (e.g., 'Circular Queue' vs 'Priority Queue' vs 'Linear Queue').",
                 "Level 3: Dense Semantic Vector Search: nomic-embed-text finds semantically identical questions phrased differently.",
                 "Level 4: Cross-Subject Fallback: If a student asks a Data Structures question with 'Machine Learning' selected, the engine auto-searches all BU subjects seamlessly."
             ])

    add_card(slide7, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2),
             title="Why One FAISS Index Per Subject?",
             title_color=ACCENT_BLUE,
             items=[
                 "The Crowding Problem: In a single giant vectorstore, large subjects with 500 pages drown out small subjects with only 40 pages.",
                 "Search Precision: Subject isolation guarantees queries retrieve context exclusively from the relevant course syllabus.",
                 "Sub-Second Search: Searching a 500-chunk index takes < 5 milliseconds vs querying tens of thousands of chunks.",
                 "Modular Maintenance: Rebuilding or updating one subject's notes does not invalidate or require re-indexing other subjects."
             ])

    # =========================================================================
    # SLIDE 8: CORE APPLICATION MODULES
    # =========================================================================
    slide8 = add_blank_slide_with_bg()
    add_header(slide8, "Application Modules & User Interface")

    add_card(slide8, Inches(0.8), Inches(1.6), Inches(5.6), Inches(2.45),
             title="1. Intelligent Chat Assistant",
             title_color=ACCENT_BLUE,
             items=[
                 "Two-part answer: Textbook grounded definition + clear analogy.",
                 "Exam intelligence widget: Verbatim past questions, years, & marks.",
                 "Lab program generator: Produces standard C code matching lab manual."
             ])

    add_card(slide8, Inches(6.8), Inches(1.6), Inches(5.7), Inches(2.45),
             title="2. Exam Intelligence Dashboard",
             title_color=ACCENT_GREEN,
             items=[
                 "Subject Analytics: Identifies high-weightage topics and marks distribution.",
                 "Priority Tiering: Categorizes topics into Tier 1 (Must-Study) vs Tier 3.",
                 "Trend Analysis: Detects recurring questions across exam sittings."
             ])

    add_card(slide8, Inches(0.8), Inches(4.35), Inches(5.6), Inches(2.45),
             title="3. Solved Question Papers",
             title_color=ACCENT_AMBER,
             items=[
                 "Bundled Solutions: Pre-computed model solutions for the most repeated questions.",
                 "Curated + Model Generated: High-quality manual answers merged with LLM output.",
                 "Exportable: One-click export to printable text/PDF format."
             ])

    add_card(slide8, Inches(6.8), Inches(4.35), Inches(5.7), Inches(2.45),
             title="4. Admin Setup & Corpus Management",
             title_color=ACCENT_INDIGO,
             items=[
                 "Document Ingestion: Upload new PDFs and syllabi through the UI.",
                 "Automatic OCR Pipeline: Scanned files automatically converted to searchable text.",
                 "Instant Question Bank Rebuild: Rebuilds question index in seconds without re-indexing notes."
             ])

    # =========================================================================
    # SLIDE 9: EVALUATION & RIGOROUS TESTING RESULTS
    # =========================================================================
    slide9 = add_blank_slide_with_bg()
    add_header(slide9, "Evaluation, Verification & Testing Results")

    add_card(slide9, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             title="Automated Test Harness (eval_questions.py)",
             title_color=ACCENT_BLUE,
             items=[
                 "29 Regression Test Cases: Executed end-to-end through the exact same pipeline used by the Streamlit chat interface.",
                 "44 Programmatic Verification Checks: Automated PASS / FAIL / WARN assertions per query.",
                 "Topic Carryover Guard: Verifies that follow-up questions do not bleed unrelated previous topics.",
                 "Sitting Count Consistency: Asserts that reported sitting counts match listed exam years exactly.",
                 "Lab Code Verification: Ensures requested lab programs strictly produce compilable C code."
             ])

    add_card(slide9, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2),
             title="Key Quantitative Results",
             title_color=ACCENT_GREEN,
             items=[
                 "Question Bank Coverage: 1,426 clean exam questions extracted across 6 MCA semesters.",
                 "Out-of-Syllabus Rejection: 100% rejection rate for non-syllabus topics in 0.00 seconds.",
                 "Deterministic Accuracy: 0% calculation hallucination for past exam frequency counts.",
                 "Answer Cache Hit Rate: Instant < 0.10 second latency for repeated student queries.",
                 "Memory Footprint: Stable ~4.1 GB RAM consumption throughout continuous testing."
             ])

    # =========================================================================
    # SLIDE 10: STEP-BY-STEP LIVE DEMO GUIDE
    # =========================================================================
    slide10 = add_blank_slide_with_bg()
    add_header(slide10, "Live Demonstration Plan (For Examiners)")

    add_card(slide10, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.2),
             title="Step 1: Conceptual Query",
             title_color=ACCENT_BLUE,
             items=[
                 "Prompt: 'Explain B-Tree with an analogy'",
                 "What to Show:",
                 "1. System retrieves top-3 notes chunks from Data Structures.",
                 "2. Part 1: Academic textbook explanation with properties.",
                 "3. Part 2: Intuitive analogy (multi-lane highway / library index).",
                 "4. Exact past BU questions shown with marks."
             ])

    add_card(slide10, Inches(4.8), Inches(1.6), Inches(3.6), Inches(5.2),
             title="Step 2: Exam Stats Query",
             title_color=ACCENT_GREEN,
             items=[
                 "Prompt: 'How many times has Dijkstra\\'s algorithm been asked?'",
                 "What to Show:",
                 "1. Python engine identifies algorithm in question bank.",
                 "2. Instant deterministic response (e.g. 'Asked 4 times: 2022, 2023, 2024, 2025').",
                 "3. Displays exact marks breakdown (5M vs 10M).",
                 "4. Zero LLM calculation hallucination."
             ])

    add_card(slide10, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.2),
             title="Step 3: Guardrail & Stats",
             title_color=ACCENT_AMBER,
             items=[
                 "Prompt: 'Explain Quantum Computing'",
                 "What to Show:",
                 "1. Syllabus gate checks query against MCA syllabus.",
                 "2. 0.00s instant refusal: 'This topic is outside the Bangalore University MCA syllabus.'",
                 "Tab Demo: Exam Intelligence",
                 "1. High-weightage topics per subject.",
                 "2. One-click solution export."
             ])

    # =========================================================================
    # SLIDE 11: VIVA DEFENSE — TOP QUESTIONS & ANSWERS
    # =========================================================================
    slide11 = add_blank_slide_with_bg()
    add_header(slide11, "Anticipated Viva Defense Questions & Answers")

    add_card(slide11, Inches(0.8), Inches(1.6), Inches(5.6), Inches(2.45),
             title="Q1: Why RAG instead of Fine-Tuning?",
             title_color=ACCENT_BLUE,
             items=[
                 "Answer: Fine-tuning alters model weights, requires GPU clusters, suffers from catastrophic forgetting, and cannot cite sources.",
                 "RAG Advantages: Dynamic updates (just drop a new PDF), zero hallucination of factual notes, and exact page citations."
             ])

    add_card(slide11, Inches(6.8), Inches(1.6), Inches(5.7), Inches(2.45),
             title="Q2: How do you prevent hallucinations?",
             title_color=ACCENT_GREEN,
             items=[
                 "Answer: 1) Strict grounding prompt ('Answer ONLY from retrieved context').",
                 "2) Deterministic code handles all numbers and counts—LLM never counts.",
                 "3) Hard rejection gate if syllabus/question bank match is below threshold."
             ])

    add_card(slide11, Inches(0.8), Inches(4.35), Inches(5.6), Inches(2.45),
             title="Q3: What is the 7B in Mistral 7B?",
             title_color=ACCENT_AMBER,
             items=[
                 "Answer: 7 Billion parameters (weights). At FP16 precision, it takes 14 GB RAM.",
                 "Quantization: Using 4-bit GGUF quantization (Q4_K_M) compresses weights to 4.1 GB, allowing smooth execution on a 16 GB laptop CPU."
             ])

    add_card(slide11, Inches(6.8), Inches(4.35), Inches(5.7), Inches(2.45),
             title="Q4: How do you extract scanned years?",
             title_color=ACCENT_INDIGO,
             items=[
                 "Answer: Multi-year papers are merged into single PDFs. Tesseract OCR reads exam headers (e.g. 'Examination, June 2024') using regex.",
                 "State Retention: Extracted year is carried forward across continuation pages until a new exam header is encountered."
             ])

    # =========================================================================
    # SLIDE 12: CONCLUSION & FUTURE WORK
    # =========================================================================
    slide12 = add_blank_slide_with_bg()
    add_header(slide12, "Conclusion & Future Enhancements")

    add_card(slide12, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2),
             title="Key Accomplishments",
             title_color=ACCENT_GREEN,
             items=[
                 "100% Offline AI: Practical, fully offline university exam study assistant running on consumer laptop hardware.",
                 "Dual-Engine Design: Successfully combines the expressive power of LLMs with the mathematical certainty of deterministic code.",
                 "Syllabus Grounding: Eliminates false knowledge and restricts answers strictly to verified course literature.",
                 "Zero Operational Cost: Zero cloud dependencies, zero subscription costs, and 100% student data privacy.",
                 "Comprehensive Verification: Backed by 29 automated test cases and 1,426 indexed university questions."
             ])

    add_card(slide12, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2),
             title="Future Roadmap",
             title_color=ACCENT_INDIGO,
             items=[
                 "Multi-Modal Vision RAG: Direct extraction and generation of complex data structures and architecture diagrams.",
                 "Multi-University Multi-Tenancy: Extending configuration to support VTU, Anna University, and other curricula seamlessly.",
                 "Voice Viva Simulator: Adding offline Whisper speech-to-text for interactive oral viva practice.",
                 "Cross-Subject Knowledge Graph: Visualizing prerequisite concept dependencies across semesters."
             ])

    # =========================================================================
    # SLIDE 13: THANK YOU & Q&A
    # =========================================================================
    slide13 = add_blank_slide_with_bg()

    t_box13 = slide13.shapes.add_textbox(Inches(1.5), Inches(2.0), Inches(10.333), Inches(3.5))
    tf13 = t_box13.text_frame
    tf13.word_wrap = True

    p_th = tf13.paragraphs[0]
    p_th.alignment = PP_ALIGN.CENTER
    p_th.text = "Thank You!"
    p_th.font.size = Pt(54)
    p_th.font.bold = True
    p_th.font.color.rgb = ACCENT_BLUE

    p_sub = tf13.add_paragraph()
    p_sub.alignment = PP_ALIGN.CENTER
    p_sub.text = "Open for Viva Defense & Questions"
    p_sub.font.size = Pt(24)
    p_sub.font.bold = True
    p_sub.font.color.rgb = TEXT_MAIN
    p_sub.space_before = Pt(14)

    p_det = tf13.add_paragraph()
    p_det.alignment = PP_ALIGN.CENTER
    p_det.text = (
        "Raghavendra Chandrakantha Bhat | Reg No: P03NK24S126033\n"
        "Master of Computer Applications (MCA) | Bangalore University\n"
        "Guide: Dr. Muralidhara B.L."
    )
    p_det.font.size = Pt(16)
    p_det.font.color.rgb = TEXT_MUTED
    p_det.space_before = Pt(16)

    output_path = os.path.abspath("uniRAG_Lab_Exam_Presentation.pptx")
    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")

if __name__ == "__main__":
    create_deck()
