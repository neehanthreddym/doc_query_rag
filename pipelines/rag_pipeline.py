from typing import Dict, Any, List
from langchain_groq import ChatGroq
from core.retriever import RAGRetriever

class RAGPipeline:
    """Encapsulates the full RAG pipeline from query to answer."""
    
    def __init__(self, retriever: RAGRetriever, llm: ChatGroq):
        self.retriever = retriever
        self.llm = llm

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Formats the retrieved documents into a single context string."""
        return "\n\n---\n\n".join([doc['content'] for doc in documents])

    def _create_prompt(self, query: str, context: str) -> str:
        """Creates the final prompt for the LLM."""
        return f"""You are an expert research Q&A assistant. Your task is to use the following context to answer the question.

        Instructions:
        1. Do not make assumptions.
        2. Elaborate your answers with details from the context.
        3. Cite sources when applicable.
        4. If multiple sources provide the same information, consolidate them into a single answer.
        5. If the context does not contain enough information to answer the query, reply with: "I cannot answer this query based on the provided context."
        6. Produce answers in markdown format for better readability.

        Here is the context and the question you need to answer:
        ---
        CONTEXT:
        {context}
        ---
        QUESTION:
        {query}
        ---
        ANSWER:
        """

    def ask(self, query: str, top_k: int = 3, min_score: float = 0.2) -> Dict[str, Any]:
        """Executes the RAG pipeline."""
        # 1. Retrieve documents
        retrieved_docs = self.retriever.retrieve(
            query=query, 
            top_k=top_k, 
            score_threshold=min_score
        )

        if not retrieved_docs:
            return {
                'answer': "No relevant context was found to answer your query.",
                'sources': [],
                'confidence': 0.0
            }

        # 2. Format context and create prompt
        context = self._format_context(retrieved_docs)
        prompt = self._create_prompt(query, context)
        
        # 3. Generate answer
        print("💬 Generating answer with LLM...")
        response = self.llm.invoke(prompt)
        
        # 4. Compile results
        sources = [{
            'source': doc['metadata'].get('source', 'unknown'),
            'page': doc['metadata'].get('page', 'N/A'),
            'score': doc['score']
        } for doc in retrieved_docs]
        
        confidence = max(doc['score'] for doc in retrieved_docs)
        
        result = {
            'answer': response.content,
            'sources': sources,
            'confidence': confidence
        }
        
        return result