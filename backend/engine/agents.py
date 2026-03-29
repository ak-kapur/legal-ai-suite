import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

class LegalAgents:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=self.api_key,
            temperature=0.1 
        )

    # --- THE STABLE PDF VAULT Q&A (Now using Modern LCEL) ---
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
        
        # This formats our retrieved documents into a single readable string
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
            
        # THE MODERN LCEL CHAIN (No old imports needed!)
        rag_chain = (
            {
                "context": retriever | format_docs, 
                "question": RunnablePassthrough(), 
                "chat_history": lambda _: chat_history # Injects the history directly
            }
            | prompt
            | self.llm
            | StrOutputParser() # Extracts just the clean text automatically!
        )
        
        # Run it!
        return rag_chain.invoke(question)

    # --- THE NEW EXPLICIT WEB SEARCH FUNCTION ---
    # --- THE HYBRID WEB SEARCH FUNCTION ---
    def search_kanoon_direct(self, retriever, question, chat_history):
        # STEP 1: Read the PDF to understand what "this case" is
        # We grab the top 3 most relevant chunks from the local vault
        pdf_docs = retriever.invoke(question)
        pdf_context = "\n\n".join(doc.page_content for doc in pdf_docs)

        # STEP 2: Smart Web Query Generator
        query_prompt = f"""Based on this excerpt from our client's case:
        {pdf_context}
        
        The user is asking: "{question}"
        
        Write a highly optimized, 3 to 6 word search query to find similar legal precedents on Indian Kanoon. 
        Focus strictly on the core legal principles. Respond with ONLY the search keywords (no quotes)."""
        
        optimized_query = self.llm.invoke(query_prompt).content.strip()
        print(f"🧠 AI Generated Search Query: {optimized_query}")
        
        # STEP 3: Scrape Indian Kanoon using the smart keywords
        search = DuckDuckGoSearchRun()
        web_results = search.invoke(f"{optimized_query} site:indiankanoon.org")
        
        # STEP 4: The Ultimate Hybrid Synthesis
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

# --- THE NEW TIMELINE GENERATOR ---
    # --- THE UPGRADED TIMELINE GENERATOR ---
    # --- THE PERFECT TIMELINE GENERATOR ---
    def generate_timeline(self, retriever):
        # 1. THE HACK: Override the default limits! 
        # We tell the database to give us up to 50 chunks instead of just 4.
        # This effectively feeds the entire document to the AI.
        retriever.search_kwargs = {"k": 50}
        
        # We use a "dumb" keyword search packed with months to force it to grab the date chunks
        docs = retriever.invoke("January February March April May June July August September October November December 2020 2021 2022 2023 2024 dates events")
        pdf_context = "\n\n".join(doc.page_content for doc in docs)
        
        # Reset the retriever back to normal for the rest of the app so Q&A stays fast
        retriever.search_kwargs = {"k": 4}
        
        # 2. Ultra-Strict Prompt Engineering
        prompt = f"""You are an elite Legal Assistant.
        Read the following comprehensive case excerpts and extract a complete, chronological timeline.
        
        CASE EXCERPTS (Full Document Context):
        {pdf_context}
        
        INSTRUCTIONS:
        1. Scan the text meticulously for ANY exact dates, months, or years.
        2. List every factual event in chronological order (oldest to newest).
        3. Format strictly as a bulleted list: **[Exact Date/Year]** - [Action/Event]
        4. If a date is explicitly mentioned in the text (e.g., "On August 10, 2024"), you MUST extract it. Do not skip dates.
        5. Ignore general legal philosophy. Focus ONLY on the factual actions of the parties.
        """
        
        # 3. Generate and return
        return self.llm.invoke(prompt).content