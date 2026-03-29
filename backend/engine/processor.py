from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, 
            chunk_overlap=150
        )

    def process_to_documents(self, pdf_files):
        all_docs = []
        for pdf in pdf_files:
            # For FastAPI UploadFile, use .filename and .file
            filename = pdf.filename 
            try:
                reader = PdfReader(pdf.file) # Read the file stream directly
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        doc = Document(
                            page_content=text,
                            metadata={"source": filename, "page": i + 1}
                        )
                        all_docs.extend(self.splitter.split_documents([doc]))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        return all_docs