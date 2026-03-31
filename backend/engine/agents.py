import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

class LegalAgents:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self._llm = None  # Lazy-loading placeholder

    def _clean_response(self, response):
        """
        UNIVERSAL 2026 SCRUBBER: 
        Extracts raw text and deletes Gemini 3's metadata/signatures.
        """
        # Step 1: Extract the content (could be a string or a list of parts)
        # If passed a string (e.g. from StrOutputParser), return it directly
        if isinstance(response, str):
            return response

        # Extract content from the AIMessage object
        content = getattr(response, 'content', response)
        
        # Step 2: Handle the 2026 List-of-Blocks format
        if isinstance(content, list):
            # Extract only the 'text' field from the first dictionary block
            first_part = content[0]
            if isinstance(first_part, dict):
                return first_part.get('text', str(first_part))
            return str(first_part)
        
        # Step 3: Handle standard string content
        return str(content)

    @property
    def llm(self):
        """Lazy-loads Gemini 3 Flash to ensure fast boot and low RAM usage."""
        if self._llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            if not self.api_key:
                print("⚠️ WARNING: GOOGLE_API_KEY is missing in Environment Variables!")
            
            self._llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview", 
                google_api_key=self.api_key,
                temperature=0.1 
            )
        return self._llm

    # --- PDF VAULT Q&A ---
    def get_lawyer_response(self, retriever, question, chat_history):
        template = """ROLE: Senior Legal Counsel.
        CONTEXT: {context}
        HISTORY: {chat_history}
        INSTRUCTIONS: Be persuasive. Cite page numbers. Reference history.
        QUESTION: {question}
        STRATEGIC ADVICE:"""
        return self._run_chain(retriever, template, question, chat_history)

    def get_judge_response(self, retriever, question, chat_history):
        template = """ROLE: Honorable Judge.
        CONTEXT: {context}
        HISTORY: {chat_history}
        INSTRUCTIONS: Neutral tone. Reference page numbers.
        QUESTION: {question}
        LEGAL OPINION:"""
        return self._run_chain(retriever, template, question, chat_history)

    def _run_chain(self, retriever, template, question, chat_history):
        prompt = PromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
            
        rag_chain = (
            {
                "context": retriever | format_docs, 
                "question": RunnablePassthrough(), 
                "chat_history": lambda _: chat_history 
            }
            | prompt
            | self.llm
            | StrOutputParser() # This usually returns a string, but we scrub it anyway below
        )
        
        raw_ans = rag_chain.invoke(question)
        return self._clean_response(raw_ans)

    # --- INDIAN KANOON WEB SEARCH ---
    def search_kanoon_direct(self, retriever, question, chat_history):
        pdf_docs = retriever.invoke(question)
        pdf_context = "\n\n".join(doc.page_content for doc in pdf_docs)

        # 1. Create Query
        query_prompt = f"""Extract 3 to 6 keywords for a legal search on Indian Kanoon based on:
        {pdf_context}
        QUESTION: {question}
        QUERY:"""
        
        # FIX: Scrub the query BEFORE stripping it to avoid 'list' has no attribute 'strip'
        raw_query = self.llm.invoke(query_prompt)
        optimized_query = self._clean_response(raw_query).strip().replace('"', '')
        
        # 2. Execute Web Search
        search = DuckDuckGoSearchRun()
        print(f"--- Searching Indian Kanoon for: {optimized_query} ---")
        web_results = search.invoke(f"{optimized_query} site:indiankanoon.org")
        
        # 3. Formulate Final Response
        final_prompt = f"""You are a Senior Legal Counsel.
        CASE FACTS: {pdf_context}
        WEB PRECEDENTS: {web_results}
        USER QUESTION: {question}
        COMPARE AND ADVISE:"""
        
        raw_final = self.llm.invoke(final_prompt)
        return self._clean_response(raw_final)

    # --- CHRONOLOGICAL TIMELINE ---
    def generate_timeline(self, retriever):
        # Broad Retrieval
        retriever.search_kwargs = {"k": 50}
        docs = retriever.invoke("dates and events")
        pdf_context = "\n\n".join(doc.page_content for doc in docs)
        retriever.search_kwargs = {"k": 5} # Reset
        
        prompt = f"""Extract a chronological timeline.
        CASE EXCERPTS: {pdf_context}
        FORMAT: **[Exact Date]** - [Event]"""
        
        raw_response = self.llm.invoke(prompt)
        return self._clean_response(raw_response)