import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

class LegalAgents:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self._llm = None  # We start with None to boot instantly

    @property
    def llm(self):
        """Lazy-loads the LLM only when a question is actually asked."""
        if self._llm is None:
            # Import inside the property to save boot time
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            if not self.api_key:
                print("⚠️ WARNING: GOOGLE_API_KEY is missing in Environment Variables!")
            
            self._llm = ChatGoogleGenerativeAI(
                model="models/gemini-1.5-flash", # Note: use 1.5-flash as 2.5 isn't a standard public tier name yet
                google_api_key=self.api_key,
                temperature=0.1 
            )
        return self._llm

    # --- THE STABLE PDF VAULT Q&A ---
    def get_lawyer_response(self, retriever, question, chat_history):
        template = """ROLE: You are a Senior Legal Counsel representing a client.
        CONTEXT: Use the following case details:
        {context}
        
        PREVIOUS CONVERSATION:
        {chat_history}
        
        INSTRUCTIONS:
        1. Be persuasive. 
        2. Cite specific sources and page numbers.
        3. Use the PREVIOUS CONVERSATION for context.

        QUESTION: {question}
        STRATEGIC ADVICE:"""
        return self._run_chain(retriever, template, question, chat_history)

    def get_judge_response(self, retriever, question, chat_history):
        template = """ROLE: You are an Honorable Judge.
        CONTEXT: Use the following legal excerpts.
        {context}
        
        PREVIOUS CONVERSATION:
        {chat_history}
        
        INSTRUCTIONS: Maintain a neutral tone and reference specific pages.
        QUESTION: {question}
        LEGAL OPINION:"""
        return self._run_chain(retriever, template, question, chat_history)

    def _run_chain(self, retriever, template, question, chat_history):
        prompt = PromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
            
        # Use self.llm (the property) to trigger lazy loading
        rag_chain = (
            {
                "context": retriever | format_docs, 
                "question": RunnablePassthrough(), 
                "chat_history": lambda _: chat_history 
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain.invoke(question)

    def search_kanoon_direct(self, retriever, question, chat_history):
        pdf_docs = retriever.invoke(question)
        pdf_context = "\n\n".join(doc.page_content for doc in pdf_docs)

        query_prompt = f"""Based on this excerpt from our client's case:
        {pdf_context}
        
        The user is asking: "{question}"
        
        Write a highly optimized, 3 to 6 word search query to find similar legal precedents on Indian Kanoon. 
        Focus strictly on the core legal principles. Respond with ONLY the search keywords (no quotes)."""
        
        optimized_query = self.llm.invoke(query_prompt).content.strip()
        
        search = DuckDuckGoSearchRun()
        web_results = search.invoke(f"{optimized_query} site:indiankanoon.org")
        
        final_prompt = f"""You are a Senior Legal Counsel.
        
        OUR CASE FACTS (From the Uploaded PDF):
        {pdf_context}
        
        INDIAN KANOON PRECEDENTS FOUND FROM WEB:
        {web_results}
        
        USER QUESTION: {question}
        
        INSTRUCTIONS: 
        1. Compare the Indian Kanoon precedents directly to our case facts.
        2. Point out similarities or differences.
        3. Cite the case names/details from the web results if available.
        """
        return self.llm.invoke(final_prompt).content

    def generate_timeline(self, retriever):
        retriever.search_kwargs = {"k": 50}
        docs = retriever.invoke("dates events 2023 2024 2025")
        pdf_context = "\n\n".join(doc.page_content for doc in docs)
        retriever.search_kwargs = {"k": 4}
        
        prompt = f"""You are an elite Legal Assistant. Extract a chronological timeline.
        
        CASE EXCERPTS:
        {pdf_context}
        
        FORMAT: **[Exact Date]** - [Event]
        """
        return self.llm.invoke(prompt).content