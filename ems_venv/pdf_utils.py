from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap

def create_pdf_from_content(text_content: str, pdf_path: str):
    """
    Generate a PDF file from plain text content.
    Mermaid diagrams will be included as raw text.
    """
    wrapper = textwrap.TextWrapper(width=90)
    wrapped_text = wrapper.wrap(text_content)

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    text_obj = c.beginText(40, height - 40)
    text_obj.setFont("Courier", 10)

    for line in wrapped_text:
        text_obj.textLine(line)
        # New page if text overflows
        if text_obj.getY() < 40:
            c.drawText(text_obj)
            c.showPage()
            text_obj = c.beginText(40, height - 40)
            text_obj.setFont("Courier", 10)

    c.drawText(text_obj)
    c.save()
