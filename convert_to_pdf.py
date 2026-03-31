import markdown
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfgen import canvas
import re

def markdown_to_pdf(md_file, pdf_file):
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='#1a1a1a',
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor='#2c3e50',
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor='#34495e',
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontSize=12,
        textColor='#7f8c8d',
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        textColor='#000000',
        fontName='Helvetica',
        leading=14
    )
    
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Normal'],
        fontSize=9,
        textColor='#e74c3c',
        fontName='Courier',
        backColor='#f5f5f5',
        leftIndent=20,
        rightIndent=20,
        leading=12
    )
    
    # Process markdown content line by line
    lines = md_content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if not line:
            story.append(Spacer(1, 0.1*inch))
            continue
        
        # Handle headers
        if line.startswith('# '):
            text = line[2:].strip()
            story.append(Paragraph(text, title_style))
            story.append(Spacer(1, 0.2*inch))
        elif line.startswith('## '):
            text = line[3:].strip()
            story.append(Paragraph(text, heading1_style))
            story.append(Spacer(1, 0.15*inch))
        elif line.startswith('### '):
            text = line[4:].strip()
            story.append(Paragraph(text, heading2_style))
            story.append(Spacer(1, 0.1*inch))
        elif line.startswith('#### '):
            text = line[5:].strip()
            story.append(Paragraph(text, heading3_style))
            story.append(Spacer(1, 0.05*inch))
        
        # Handle code blocks (simple detection)
        elif line.startswith('```') or line.startswith('import ') or line.startswith('def ') or \
             line.startswith('from ') or line.startswith('#') or line.startswith('if ') or \
             line.startswith('    ') or '=' in line and ('pickle' in line or 'model' in line or 'df' in line):
            # Skip code block markers
            if not line.startswith('```'):
                code_text = line.replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(code_text, code_style))
        
        # Handle table headers (| syntax)
        elif line.startswith('|') and line.endswith('|'):
            # Convert table row to paragraph
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            table_text = " | ".join(cells)
            story.append(Paragraph(table_text, normal_style))
        
        # Handle bullet points
        elif line.startswith('- ') or line.startswith('* '):
            text = line[2:].strip()
            story.append(Paragraph(f"• {text}", normal_style))
        
        # Handle regular text
        else:
            # Clean up markdown syntax
            text = line
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)  # Italic
            text = re.sub(r'`([^`]+)`', r'<font color="#e74c3c" face="Courier">\1</font>', text)  # Inline code
            
            if text.strip():
                story.append(Paragraph(text, normal_style))
        
        # Add page break marker handling
        if '---' in line:
            story.append(PageBreak())
    
    # Build PDF
    doc.build(story)
    print(f"[OK] PDF created successfully: {pdf_file}")

if __name__ == "__main__":
    markdown_to_pdf('ML_MODEL_DOCUMENTATION.md', 'ML_MODEL_DOCUMENTATION.pdf')
