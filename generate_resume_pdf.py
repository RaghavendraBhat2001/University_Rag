import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_resume(output_pdf="Raghavendra_Chandrakantha_Bhat_Resume.pdf"):
    # Target exact 1 page with standard 0.5 in margins (36 pt)
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32
    )

    styles = getSampleStyleSheet()

    # Color Palette - Professional ATS Navy & Charcoal
    COLOR_PRIMARY = colors.HexColor("#0F172A")    # Deep Slate / Navy
    COLOR_BLUE = colors.HexColor("#0284C7")       # Accent Blue (Sky 600)
    COLOR_DARK_BLUE = colors.HexColor("#1E3A8A")  # Royal Navy
    COLOR_TEXT = colors.HexColor("#1E293B")       # Slate 800
    COLOR_MUTED = colors.HexColor("#475569")      # Slate 600
    COLOR_BORDER = colors.HexColor("#CBD5E1")     # Slate 300

    # Typography Styles
    name_style = ParagraphStyle(
        'ResumeName',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=23,
        alignment=1, # Center
        textColor=COLOR_PRIMARY
    )

    contact_style = ParagraphStyle(
        'ResumeContact',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.8,
        leading=13,
        alignment=1,
        textColor=COLOR_TEXT
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=COLOR_DARK_BLUE,
        spaceBefore=7,
        spaceAfter=2
    )

    summary_style = ParagraphStyle(
        'SummaryText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.8,
        leading=12.2,
        textColor=COLOR_TEXT,
        alignment=4 # Justified
    )

    item_title_left = ParagraphStyle(
        'ItemTitleLeft',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.2,
        leading=12.5,
        textColor=COLOR_PRIMARY
    )

    item_title_right = ParagraphStyle(
        'ItemTitleRight',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12.5,
        alignment=2, # Right aligned
        textColor=COLOR_MUTED
    )

    bullet_style = ParagraphStyle(
        'ResumeBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.6,
        leading=11.6,
        textColor=COLOR_TEXT,
        leftIndent=11,
        firstLineIndent=-11
    )

    skill_category = ParagraphStyle(
        'SkillCategory',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.8,
        leading=12.5,
        textColor=COLOR_PRIMARY
    )

    story = []

    # 1. NAME HEADER
    story.append(Paragraph("RAGHAVENDRA CHANDRAKANTHA BHAT", name_style))
    story.append(Spacer(1, 3))

    # 2. CONTACT BAR (With Portfolio & GitHub)
    contact_line_1 = (
        "Bangalore, India &nbsp;|&nbsp; "
        "<b>+91 7996993937</b> &nbsp;|&nbsp; "
        "<font color='#0284c7'><u><a href='mailto:Raghavendra.c.bhat@gmail.com'>Raghavendra.c.bhat@gmail.com</a></u></font>"
    )
    contact_line_2 = (
        "<b>Portfolio:</b> <font color='#0284c7'><u><a href='https://raghavendrabhat2001.github.io/portfolio/'>raghavendrabhat2001.github.io/portfolio</a></u></font> &nbsp;|&nbsp; "
        "<b>LinkedIn:</b> <font color='#0284c7'><u><a href='https://linkedin.com/in/raghavendra-bhat-6182a0384'>raghavendra-bhat-6182a0384</a></u></font> &nbsp;|&nbsp; "
        "<b>GitHub:</b> <font color='#0284c7'><u><a href='https://github.com/RaghavendraBhat2001'>RaghavendraBhat2001</a></u></font>"
    )
    story.append(Paragraph(contact_line_1, contact_style))
    story.append(Paragraph(contact_line_2, contact_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_DARK_BLUE, spaceBefore=2, spaceAfter=4))

    # 3. PROFESSIONAL SUMMARY
    story.append(Paragraph("PROFESSIONAL SUMMARY", section_heading))
    summary_text = (
        "Motivated MCA student with solid software engineering fundamentals in <b>Python, Java, and SQL</b>, alongside "
        "hands-on experience developing production-grade <b>Retrieval-Augmented Generation (RAG)</b> systems and end-to-end "
        "Machine Learning pipelines. Proven ability in building offline local LLM workflows, FAISS vector similarity search, and automated "
        "data extraction pipelines. Seeking an entry-level <b>Software Developer / SDE, AI/ML Engineer, or Backend Developer</b> role."
    )
    story.append(Paragraph(summary_text, summary_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_BORDER, spaceBefore=2, spaceAfter=4))

    # 4. TECHNICAL SKILLS
    story.append(Paragraph("TECHNICAL SKILLS", section_heading))
    
    skills_data = [
        [
            Paragraph("<b>Languages:</b>", skill_category),
            Paragraph("Python, Java, SQL, C, HTML5, CSS3, JavaScript", summary_style)
        ],
        [
            Paragraph("<b>AI & GenAI:</b>", skill_category),
            Paragraph("Retrieval-Augmented Generation (RAG), LangChain, FAISS Vector Search, Ollama, Vector Embeddings", summary_style)
        ],
        [
            Paragraph("<b>ML & Data:</b>", skill_category),
            Paragraph("Scikit-learn, Pandas, NumPy, Data Preprocessing, Model Evaluation, Random Forest", summary_style)
        ],
        [
            Paragraph("<b>Databases & Tools:</b>", skill_category),
            Paragraph("MySQL, Streamlit, Git/GitHub, PyMuPDF, Tesseract OCR, VS Code, Eclipse", summary_style)
        ]
    ]
    skills_table = Table(skills_data, colWidths=[110, 430])
    skills_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
    ]))
    story.append(skills_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_BORDER, spaceBefore=2, spaceAfter=4))

    # 5. EDUCATION
    story.append(Paragraph("EDUCATION", section_heading))

    edu_data_1 = [
        [
            Paragraph("<b>Master of Computer Applications (MCA)</b> &nbsp;|&nbsp; Bangalore University, Bengaluru", item_title_left),
            Paragraph("2024 – 2026 (Expected)", item_title_right)
        ]
    ]
    t_edu1 = Table(edu_data_1, colWidths=[400, 140])
    t_edu1.setStyle(TableStyle([('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,-1), 1)]))
    story.append(t_edu1)

    edu_data_2 = [
        [
            Paragraph("<b>Bachelor of Computer Applications (BCA)</b> &nbsp;|&nbsp; Vasavi Jnana Peetha Degree College, Bengaluru", item_title_left),
            Paragraph("2021 – 2024 &nbsp;|&nbsp; <b>CGPA: 8.91</b>", item_title_right)
        ]
    ]
    t_edu2 = Table(edu_data_2, colWidths=[400, 140])
    t_edu2.setStyle(TableStyle([('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,-1), 1)]))
    story.append(t_edu2)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_BORDER, spaceBefore=2, spaceAfter=4))

    # 6. PROJECTS
    story.append(Paragraph("PROJECTS", section_heading))

    # Project 1: uniRAG
    p1_header = [
        [
            Paragraph("<b>uniRAG — University Academic RAG & Exam Intelligence System</b>", item_title_left),
            Paragraph("<i>Python, LangChain, FAISS, Ollama, Streamlit</i>", item_title_right)
        ]
    ]
    t_p1 = Table(p1_header, colWidths=[360, 180])
    t_p1.setStyle(TableStyle([('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,-1), 1)]))
    story.append(t_p1)

    p1_bullets = [
        "Built a 100% offline, privacy-first RAG pipeline using 4-bit quantized Mistral-7B and Nomic dense embeddings indexing 22 MCA subjects.",
        "Implemented a 4-level hybrid search engine (Exact, Subtopic, Vector Match, Filters) over 1,420+ past exam questions with trend analytics.",
        "Engineered an intelligent query understanding parser with deterministic syllabus-grounding guards that eliminate hallucinations in 0.00s.",
        "Developed multi-tier SHA-256 caching and marks-calibrated token budgeting (2–10 marks), slashing cached query response times to sub-second latency.",
        "Deployed an interactive Streamlit UI supporting real-time SSE token streaming, question paper PDF solvers, and role-based admin security."
    ]
    for b in p1_bullets:
        story.append(Paragraph(f"• &nbsp;{b}", bullet_style))

    story.append(Spacer(1, 3))

    # Project 2: Student Performance
    p2_header = [
        [
            Paragraph("<b>Student Performance Prediction System</b>", item_title_left),
            Paragraph("<i>Python, Scikit-learn, Pandas, Jupyter</i>", item_title_right)
        ]
    ]
    t_p2 = Table(p2_header, colWidths=[360, 180])
    t_p2.setStyle(TableStyle([('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,-1), 1)]))
    story.append(t_p2)

    p2_bullets = [
        "Trained a Random Forest classification model achieving <b>86.5% accuracy</b> in predicting student semester outcomes and academic success.",
        "Preprocessed and analyzed 500+ student records across 10+ indicators including SGPA, backlogs, attendance, and internal marks.",
        "Evaluated model performance using confusion matrices, precision, recall, and k-fold cross-validation to prevent feature overfitting."
    ]
    for b in p2_bullets:
        story.append(Paragraph(f"• &nbsp;{b}", bullet_style))

    story.append(Spacer(1, 3))

    # Project 3: Flight Delay
    p3_header = [
        [
            Paragraph("<b>Flight Delay Prediction System</b>", item_title_left),
            Paragraph("<i>Python, Scikit-learn, Machine Learning, Pickle</i>", item_title_right)
        ]
    ]
    t_p3 = Table(p3_header, colWidths=[360, 180])
    t_p3.setStyle(TableStyle([('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,-1), 1)]))
    story.append(t_p3)

    p3_bullets = [
        "Developed a predictive machine learning pipeline to forecast commercial flight departure delays based on weather data and carrier schedules.",
        "Performed exploratory data analysis (EDA), handled missing values, categorical encoding, and serialized trained models using Pickle."
    ]
    for b in p3_bullets:
        story.append(Paragraph(f"• &nbsp;{b}", bullet_style))

    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_BORDER, spaceBefore=2, spaceAfter=4))

    # 7. CERTIFICATIONS
    story.append(Paragraph("CERTIFICATIONS", section_heading))

    cert_data_1 = [
        [
            Paragraph("<b>AI Foundations Associate</b> – Oracle <i>(Generative AI, OCI AI Services, Machine Learning, Deep Learning)</i>", summary_style),
            Paragraph("Sep 2025", item_title_right)
        ]
    ]
    t_c1 = Table(cert_data_1, colWidths=[450, 90])
    t_c1.setStyle(TableStyle([('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,-1), 1)]))
    story.append(t_c1)

    cert_data_2 = [
        [
            Paragraph("<b>Web Developer</b> – NSDC / Skill India <i>(OOP Concepts, Java Application Development, Web Fundamentals)</i>", summary_style),
            Paragraph("Nov 2024", item_title_right)
        ]
    ]
    t_c2 = Table(cert_data_2, colWidths=[450, 90])
    t_c2.setStyle(TableStyle([('LEFTPADDING', (0,0), (-1,-1), 0), ('RIGHTPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,-1), 1)]))
    story.append(t_c2)

    doc.build(story)
    print(f"Generated clean ATS resume PDF: {os.path.abspath(output_pdf)}")

if __name__ == "__main__":
    generate_resume()
