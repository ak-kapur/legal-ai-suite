import os
from langchain_chroma import Chroma

class VectorEngine:
    def __init__(self):
        # We start with these as None so the server boots instantly
        self._embeddings = None 
        self.db_path = os.path.join(os.getcwd(), "chroma_db_vault")

    @property
    def embeddings(self):
        """
        Lazy-loads the embedding model only when first accessed.
        This prevents Render from timing out during startup.
        """
        if self._embeddings is None:
            # We import here so the library isn't even loaded until needed
            from langchain_huggingface import HuggingFaceEmbeddings
            print("--- Initializing AI Embedding Model (Lazy Load) ---")
            self._embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
        return self._embeddings

    def save_documents(self, documents):
        """
        Takes a list of LangChain Document objects and saves them to ChromaDB.
        """
        # Calling self.embeddings triggers the @property loader above
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        return vector_db

    def get_retriever(self):
        """
        Loads the existing database and returns a retriever object.
        """
        vector_db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )
        return vector_db.as_retriever(search_kwargs={"k": 5})