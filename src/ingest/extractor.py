import os
import pdfplumber

def extract_pdf(file_path: str) -> dict:
    """
    Extract text and tables from PDF.
    Returns dict with { 'text': str, 'pages': [page_texts], 'tables': [tables] }
    """
    text_all = ""
    pages = []
    tables = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            pages.append(page_text)
            text_all += (page_text or "") + "\n"

            # Try table extraction
            page_tables = page.extract_tables()
            if page_tables:
                tables.extend(page_tables)

    return {
        "text": text_all,
        "pages": pages,
        "tables": tables
    }

def extract_txt(file_path: str) -> dict:
    """
    Extract plain text file.
    Returns dict with only 'text'.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return {"text": text, "pages": [text], "tables": []}

def extract_docx(file_path: str) -> dict:
    """
    Extract text from a DOCX file.
    Returns dict with only 'text'.
    """
    from docx import Document
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return {"text": text, "pages": [text], "tables": []}

def extract_file(file_path: str) -> dict:
    """
    Generic dispatcher for different file types.
    Always returns a dict with { 'text': str, 'pages': list, 'tables': list }.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_pdf(file_path)
    elif ext == ".txt":
        return extract_txt(file_path)
    elif ext == ".docx":
        return extract_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
