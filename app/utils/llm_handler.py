import os
from langchain_groq import ChatGroq
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

class LLMHandler:
    def __init__(self):
        """Initialize the LLM handler with Groq model."""
        try:
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2
            )
            print("LLM initialized successfully")
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            raise Exception(f"Failed to initialize LLM: {str(e)}")
    
    async def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        temperature: float = 0
    ) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            query (str): User's question
            context (str, optional): Additional context for the LLM
            temperature (float): Temperature for response generation
            
        Returns:
            str: Generated response
        """
        try:
            # Prepare the prompt
            if context:
                prompt = f"""Context: {context}

Question: {query}

Provide a detailed answer based on the given context. If the context lacks relevant information, acknowledge its absence in a natural and concise way without using the phrase 'The context provided appears to be...'"""
            else:
                prompt = query

            # Generate response
            response = await self.llm.ainvoke(prompt)
            
            print(f"✅ Generated response for query: {query[:50]}...")
            return response.content
            
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            raise Exception(f"Failed to generate response: {str(e)}")

# Create a singleton instance
llm_handler = LLMHandler() 