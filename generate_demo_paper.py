import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.graphics.barcode import code128

def generate_single_question_paper(output_filename="Demo_Question_Paper_Data_Structures.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles matching university examination typography
    style_code = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        alignment=2 # Right aligned
    )

    style_title = ParagraphStyle(
        'UniTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=16,
        alignment=1 # Center aligned
    )

    style_sub = ParagraphStyle(
        'UniSub',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=15,
        alignment=1
    )

    style_meta = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14
    )

    style_instructions = ParagraphStyle(
        'InstructionsStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=14
    )

    style_section = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=15,
        alignment=1
    )

    style_section_desc = ParagraphStyle(
        'SectionDesc',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=14
    )

    style_question = ParagraphStyle(
        'QuestionText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=16
    )

    story = []

    # Barcode and Code header
    barcode = code128.Code128("JP768", barHeight=18, barWidth=1.1)
    
    header_table_data = [
        [barcode, Paragraph("<b>JP – 768</b>", style_code)]
    ]
    header_table = Table(header_table_data, colWidths=[200, 304])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (1,0), 'RIGHT'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 14))

    # University Exam Header
    story.append(Paragraph("I Semester M.C.A. (Two Years Course) Examination, July 2023", style_title))
    story.append(Paragraph("(CBCS) (2020 – 2021 and Onwards)", style_sub))
    story.append(Paragraph("COMPUTER SCIENCE", style_sub))
    story.append(Paragraph("1MCA6 : Data Structures", style_title))
    story.append(Spacer(1, 12))

    # Time and Max Marks table
    time_marks_data = [
        [Paragraph("Time : 3 Hours", style_meta), Paragraph("Max. Marks : 70", ParagraphStyle('RightMeta', parent=style_meta, alignment=2))]
    ]
    t_table = Table(time_marks_data, colWidths=[252, 252])
    t_table.setStyle(TableStyle([
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_table)
    story.append(Spacer(1, 8))

    # Instructions
    instructions_text = (
        "<b><i>Instructions :</i></b> <i>1) Answer any five questions from Section – A.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "2) Answer any four questions from Section – B.</i>"
    )
    story.append(Paragraph(instructions_text, style_instructions))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.gray, spaceBefore=4, spaceAfter=10))

    # Section A
    story.append(Paragraph("SECTION – A", style_section))
    story.append(Spacer(1, 6))

    sec_header_data = [
        [
            Paragraph("I. Answer any five of the following. Each question carries 6 marks.", style_section_desc),
            Paragraph("(5×6=30)", ParagraphStyle('SecMarksRight', parent=style_section_desc, alignment=2))
        ]
    ]
    sec_table = Table(sec_header_data, colWidths=[384, 120])
    sec_table.setStyle(TableStyle([
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(sec_table)
    story.append(Spacer(1, 12))

    # Single Question - formatted cleanly
    story.append(Paragraph("<b>1)</b> Define data structures. Explain the classification of data structures.", style_question))
    story.append(Spacer(1, 25))
    story.append(Paragraph("________________________________________", ParagraphStyle('Divider', parent=style_meta, alignment=1, textColor=colors.gray)))

    doc.build(story)
    print(f"Generated clean single-question PDF at: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    generate_single_question_paper()
