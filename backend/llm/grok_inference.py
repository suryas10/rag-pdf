"""
Groq LLM inference for generating responses with retrieved context.
Supports streaming responses.
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from openai import OpenAI

from typing import List, Dict, Optional, Generator, Union
import os


class GroqInference:
    """Generate responses using Groq LLM with retrieved context."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        model: str = "llama-3.1-70b-versatile",
        temperature: float = 0.2,
        max_tokens: int = 800
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key must be provided or set in GROQ_API_KEY environment variable")
        
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def build_prompt(
        self,
        query: str,
        context: str,
        intent: str = "qa",
        conversation_history: Optional[List[Dict]] = None
    ) -> List[Dict[str, str]]:
        """
        Build prompt with query, context, and conversation history.
        
        Args:
            query: User query
            context: Retrieved context chunks
            intent: Query intent
            conversation_history: Previous conversation turns
        
        Returns:
            List of message dicts for chat completion
        """
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context from documents.
        
        Guidelines:
        - Use only information from the provided context to answer questions
        - If the context doesn't contain enough information, say so
        - Cite specific page numbers or sections when referencing the document
        - Be concise and accurate
        - If asked about something not in the context, politely decline"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if available
        if conversation_history:
            for turn in conversation_history[-5:]:  # Last 5 turns
                messages.append({"role": "user", "content": turn.get("query", "")})
                messages.append({"role": "assistant", "content": turn.get("response", "")})
        
        # Add context and query
        user_content = f"""Context from document:
{context}

Question: {query}

Please answer the question based on the context above. Include citations when referencing specific parts of the document."""
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def generate(
        self,
        query: str,
        context: str,
        intent: str = "qa",
        conversation_history: Optional[List[Dict]] = None,
        stream: bool = True
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate response for query with context.
        
        Args:
            query: User query
            context: Retrieved context chunks
            intent: Query intent
            conversation_history: Previous conversation turns
            stream: Whether to stream the response
        
        Returns:
            Response string or generator of chunks
        """
        messages = self.build_prompt(query, context, intent, conversation_history)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream
            )
            
            if stream:
                return self._stream_response(response)
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _stream_response(self, response) -> Generator[str, None, None]:
        """Stream response chunks."""
        for chunk in response:
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def generate_with_sources(
        self,
        query: str,
        context_chunks: List[Dict],
        intent: str = "qa",
        conversation_history: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict:
        """
        Generate response with source citations.
        
        Args:
            query: User query
            context_chunks: List of retrieved chunks with metadata
            intent: Query intent
            conversation_history: Previous conversation turns
            stream: Whether to stream the response
        
        Returns:
            Dict with response, sources, and metadata
        """
        # Format context with source information
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            page_no = metadata.get("page_no", "?")
            chunk_id = metadata.get("chunk_id", "")
            score = chunk.get("score", 0.0)
            
            context_parts.append(
                f"[Source {i} - Page {page_no}, Relevance: {score:.2f}]\n{text}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Generate response
        if stream:
            response_stream = self.generate(
                query, context, intent, conversation_history, stream=True
            )
            return {
                "response_stream": response_stream,
                "sources": context_chunks,
                "intent": intent
            }
        else:
            response = self.generate(
                query, context, intent, conversation_history, stream=False
            )
            return {
                "response": response,
                "sources": context_chunks,
                "intent": intent
            }

