import os
from langchain_chroma import Chroma

class VectorEngine:
    def __init__(self):
        # We start with None, but it will initialize instantly since it's an API call
        self._embeddings = None 
        # Using a specific path for persistent storage
        self.db_path = os.path.join(os.getcwd(), "chroma_db_vault")

    @property
    def embeddings(self):
        """
        Loads Google's Cloud Embeddings. 
        Uses 0MB of local RAM because the math happens on Google's servers.
        """
        if self._embeddings is None:
            # Import inside the property to keep the initial boot light
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            
            print("--- Initializing Google Cloud Embeddings ---")
            # Ensure your GOOGLE_API_KEY is in Render's Environment Variables
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        return self._embeddings

    def save_documents(self, documents):
        """
        Saves LangChain Document objects to ChromaDB.
        """
        # self.embeddings triggers the Cloud-based property above
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        return vector_db

    def get_retriever(self):
        """
        Loads the existing vault and returns a retriever object.
        """
        vector_db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )
        return vector_db.as_retriever(search_kwargs={"k": 5})