from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from app.models import Book, BookList
from app.scraper import fetch_book, fetch_book_content
from app.utils import get_book_content
import edge_tts
import os
# Set environment variable for tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import io
import tempfile
import asyncio
import re
from app.utils.text_chunking import chunk_text, get_chunk_info, process_text, load_embeddings
from app.utils.llm_handler import llm_handler
from app.utils.sentiment_analysis import analyze_book_sentiment
from app.utils.content_cache import clear_cache
import numpy as np
from mangum import Mangum  # Required for Vercel
import uvicorn
import time



app = FastAPI(
    title="Gutenberg API",
    description="API for accessing Project Gutenberg books",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str
    chunk_index: Optional[int] = 0
    total_chunks: Optional[int] = 1

class QueryRequest(BaseModel):
    query: str
    book_id: str
    temperature: Optional[float] = 0

def split_text_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks at sentence boundaries."""
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

@app.get("/")
async def root():
    return {"message": "Welcome to Gutenberg API"}

@app.get("/books/{book_id}", response_model=Book)
async def get_book(book_id: str):
    # Fetch directly from Gutenberg without content
    book = fetch_book(book_id, include_content=False)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book

@app.get("/books/{book_id}/content")
async def get_book_content_endpoint(book_id: str):
    print(f"\n=== Retrieving content for book {book_id} ===")
    
    # Get content from cache or fetch if needed
    content = get_book_content(book_id)
    if not content:
        print("No content found")
        raise HTTPException(status_code=404, detail="Book content not found")
    
    return content

@app.get("/books/search/{query}", response_model=List[Book])
async def search_books_route(query: str):
    # For now, return empty list since we removed the database
    return []

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        print(f"\n=== Processing TTS request ===")
        print(f"Text length: {len(request.text)} characters")
        print(f"Chunk {request.chunk_index + 1} of {request.total_chunks}")
        
        # Initialize edge-tts
        print("Initializing TTS...")
        communicate = edge_tts.Communicate(request.text, "en-US-JennyNeural", rate="+0%")
        
        # Create a temporary file
        print("Generating speech...")
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Generate speech to the temporary file
        await communicate.save(temp_path)
        
        # Read the generated audio file
        with open(temp_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        print("Successfully generated audio")
        
        # Return the audio stream with chunk information
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/mp3",
            headers={
                "Content-Disposition": f"attachment; filename=speech_chunk_{request.chunk_index}.mp3",
                "X-Chunk-Index": str(request.chunk_index),
                "X-Total-Chunks": str(request.total_chunks)
            }
        )
    except Exception as e:
        print(f"Error in TTS endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/chunks")
async def get_text_chunks(request: TTSRequest):
    """Get the text split into chunks."""
    chunks = split_text_into_chunks(request.text)
    return {
        "chunks": chunks,
        "total_chunks": len(chunks)
    }


@app.post("/books/{book_id}/ask")
async def ask_about_book(book_id: str, request: QueryRequest):
    """
    Answer questions about a specific book using the LLM.
    """
    try:
        print(f"\n=== Processing query for book {book_id} ===")
        print(f"Query: {request.query}")
        start_time = time.time()
        
        # Load existing embeddings
        chunk_embeddings = load_embeddings(book_id)
        
        # If no embeddings found, generate them
        if not chunk_embeddings:
            print(f"No embeddings found for book {book_id}, generating them now...")
            try:
                # Get content from cache or fetch if needed
                content = get_book_content(book_id)
                if not content:
                    raise HTTPException(status_code=404, detail="Book content not found")
                
                # Compile full content from pages
                full_content = "\n\n".join(content['pages'])
                
                # Process the content (chunk and generate embeddings)
                processed_content = process_text(full_content, book_id)
                chunks = processed_content["chunks"]  # This is already a list of strings
                
                print(f"✅ Generated new embeddings for book {book_id}")
            except Exception as e:
                print(f"❌ Error generating embeddings: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate embeddings. Please try again."
                )
        else:
            # Extract text from loaded embeddings
            chunks = [chunk["text"] for chunk in chunk_embeddings]
        
        # For better performance, use just a few chunks for context
        # Use first chunk (book intro) and 2 random chunks from the middle
        print(f"Using a sample of chunks for better performance")
        chunks_for_context = []
        if len(chunks) > 0:
            chunks_for_context.append(chunks[0])  # First chunk (intro)
        
        if len(chunks) > 10:
            # Add a couple chunks from the middle
            mid_idx = len(chunks) // 2
            chunks_for_context.append(chunks[mid_idx]) 
            
        # Always add the last chunk for conclusion
        if len(chunks) > 1:
            chunks_for_context.append(chunks[-1])
            
        print(f"Using {len(chunks_for_context)} chunks for context")
        context = "\n\n".join(chunks_for_context)
        
        print(f"Generating LLM response...")
        # Generate response using LLM
        response = await llm_handler.generate_response(
            query=request.query,
            context=context,
            temperature=request.temperature
        )
        
        print(f"LLM Response received: {response[:100]}...")
        
        result = {
            "query": request.query,
            "response": response,
            "book_id": book_id,
            "processing_time": f"{time.time() - start_time:.2f} seconds"
        }
        
        print(f"Returning response to frontend. Total time: {time.time() - start_time:.2f} seconds")
        return result
        
    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        # Print the full error traceback for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/books/{book_id}/sentiment", response_model=Dict[str, Any])
@app.post("/books/{book_id}/sentiment", response_model=Dict[str, Any])
async def analyze_book_sentiment_endpoint(book_id: str):
    """
    Get sentiment analysis for a book.
    Supports both GET and POST methods for flexibility.
    """
    try:
        print(f"\n=== Analyzing sentiment for book {book_id} ===")
        
        # Get book content directly using the content endpoint function
        # This is the key fix - we need the content, not just the book metadata
        content_data = get_book_content(book_id)
        
        if not content_data:
            raise HTTPException(status_code=404, detail=f"Book content for ID {book_id} not found")
        
        # Extract the full content from pages
        if "pages" not in content_data or not content_data["pages"]:
            raise HTTPException(status_code=404, detail=f"No page content available for book {book_id}")
        
        # Combine all pages into one text
        full_content = "\n\n".join(content_data["pages"])
        print(f"Content retrieved: {len(full_content)} characters")
        
        # Analyze sentiment
        result = analyze_book_sentiment(full_content, book_id)
        return result
        
    except Exception as e:
        print(f"❌ Error in sentiment endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze sentiment: {str(e)}")

@app.delete("/cache/{book_id}")
async def clear_book_cache(book_id: str):
    """
    Clear cache for a specific book.
    """
    clear_cache(book_id)
    return {"message": f"Cache cleared for book {book_id}"}

@app.delete("/cache")
async def clear_all_cache():
    """
    Clear all book content cache.
    """
    clear_cache()
    return {"message": "All book content cache cleared"}

handler = Mangum(app)  # Expose app to Vercel

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000))) 