import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

class LegalAgents:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self._llm = None  # We start with None to boot instantly

    def _clean_response(self, response):
        """Utility to strip 2026 metadata signatures."""
        content = response.content
        if isinstance(content, list):
            return content[0].get('text', str(content))
        return str(content)


    @property
    def llm(self):
        """Lazy-loads the LLM only when a question is actually asked."""
        if self._llm is None:
            # Import inside the property to save boot time
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            if not self.api_key:
                print("⚠️ WARNING: GOOGLE_API_KEY is missing in Environment Variables!")
            
            self._llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview", 
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
        # 1. Broad Retrieval
        retriever.search_kwargs = {"k": 50}
        docs = retriever.invoke("chronological events and dates")
        pdf_context = "\n\n".join(doc.page_content for doc in docs)
        retriever.search_kwargs = {"k": 5} # Reset
        
        prompt = f"""You are an elite Legal Assistant. Extract a chronological timeline.
        
        CASE EXCERPTS:
        {pdf_context}
        
        FORMAT: **[Exact Date]** - [Event]
        """
        
        # 2. Invoke the model
        raw_response = self.llm.invoke(prompt)
        
        # 3. THE 2026 SCRUBBER: 
        # Even if .content is called, it might return a list [ {'text': '...', 'extras': {...} } ]
        content = raw_response.content
        
        if isinstance(content, list):
            # If it's a list of parts, grab the text from the first part only
            first_part = content[0]
            if isinstance(first_part, dict):
                return first_part.get('text', str(first_part))
            return str(first_part)
        
        # If it's already a string, return it directly
        return str(content)        