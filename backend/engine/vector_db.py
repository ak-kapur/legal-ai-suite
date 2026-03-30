# 1. THE CRITICAL RENDER HACK: This must be at the very top!
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from langchain_chroma import Chroma

class VectorEngine:
    def __init__(self):
        # Start as None for instant server boot
        self._embeddings = None 
        # Persistent storage path in the Render project directory
        self.db_path = os.path.join(os.getcwd(), "chroma_db_vault")

    @property
    def embeddings(self):
        """
        Loads Google's Cloud Embeddings. 
        Math happens on Google's servers, saving ~400MB of local RAM.
        """
        if self._embeddings is None:
            # Lazy-import to keep the initial boot light
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            
            print("--- Initializing Google Cloud Embeddings (gemini-embedding-2-preview) ---")
            
            self._embeddings = GoogleGenerativeAIEmbeddings(
                # Using the latest model string
                model="models/gemini-embedding-2-preview",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        return self._embeddings

    def save_documents(self, documents):
        """
        Saves documents to ChromaDB using the Google Cloud Embedding API.
        """
        try:
            vector_db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.db_path
            )
            return vector_db
        except Exception as e:
            print(f"FAILED TO SAVE TO VAULT: {str(e)}")
            raise e

    def get_retriever(self):
        """
        Loads the vault and returns a retriever.
        """
        vector_db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )
        return vector_db.as_retriever(search_kwargs={"k": 5})