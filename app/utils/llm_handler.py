import os
from langchain_groq import ChatGroq
from typing import Optional
from dotenv import load_dotenv
import traceback
import json
import asyncio
from langchain_core.messages import HumanMessage
import time

# Load environment variables
load_dotenv()

# Set Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Constants
API_TIMEOUT = 30  # seconds

class LLMHandler:
    def __init__(self):
        """Initialize the LLM handler with Groq model."""
        try:
            self.llm = ChatGroq(
                model="llama-3.1-8b-instant",  # Using a smaller, faster model
                temperature=0,
                max_tokens=1000,  # Limit token generation for faster responses
                timeout=API_TIMEOUT,  # Explicit timeout
                max_retries=1  # Reduced retries to fail faster
            )
            print("LLM initialized successfully")
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            traceback.print_exc()
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
            print(f"Starting LLM response generation for query: {query[:50]}...")
            
            # Prepare the prompt
            if context:
                prompt = f"""Context: {context}

Question: {query}

Provide a concise answer based on the given context. Keep your response brief and to the point."""
            else:
                prompt = query

            print(f"Prompt prepared, sending to Groq API at {time.strftime('%H:%M:%S')}...")
            
            # Implement a timeout using asyncio
            try:
                # Create a message instead of just a string prompt
                messages = [HumanMessage(content=prompt)]
                
                # Set a timeout using asyncio
                response_task = self.llm.ainvoke(messages)
                response = await asyncio.wait_for(response_task, timeout=API_TIMEOUT)
                
                print(f"Response received from Groq API at {time.strftime('%H:%M:%S')}. Response type: {type(response)}")
                
                if not hasattr(response, 'content'):
                    print(f"WARNING: Response doesn't have 'content' attribute. Full response: {str(response)[:200]}...")
                    # Try to extract content from response if it's a dictionary
                    if isinstance(response, dict) and 'content' in response:
                        return response['content']
                    return str(response)
                
                content = response.content
                print(f"✅ Generated response: {content[:100]}...")
                return content
                
            except asyncio.TimeoutError:
                print(f"❌ Groq API request timed out after {API_TIMEOUT} seconds")
                return f"I'm sorry, but the request to the AI service timed out. Please try a simpler query or try again later."
            
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            traceback.print_exc()
            # Return a fallback response rather than raising an exception
            return f"I'm sorry, but I encountered an error while generating a response: {str(e)}. Please try again."

# Create a singleton instance
llm_handler = LLMHandler() 