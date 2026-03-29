import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class VectorEngine:
    def __init__(self):
        # No API key needed for this part! It runs locally.
        # This will download a small (~80MB) model the very first time it runs.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.db_path = "./chroma_db"

    def save_documents(self, documents):
        """
        Takes a list of LangChain Document objects (with metadata)
        and saves them to the persistent ChromaDB locally.
        """
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        return vector_db

    def get_retriever(self):
        vector_db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )
        return vector_db.as_retriever(search_kwargs={"k": 5})