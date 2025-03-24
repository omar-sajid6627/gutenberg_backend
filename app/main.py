from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from .models import Book, BookList
from .scraper import fetch_book, fetch_book_content
import edge_tts
import os
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import io
import wave
import tempfile
import asyncio
import re
from .utils.text_chunking import chunk_text, get_chunk_info, process_text, load_embeddings
from .utils.llm_handler import llm_handler
import numpy as np
from mangum import Mangum  # Required for Vercel
import uvicorn



app = FastAPI(
    title="Gutenberg API",
    description="API for accessing Project Gutenberg books",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

@app.get("/books/{book_id}", response_model=Book)
async def get_book(book_id: str):
    # Fetch directly from Gutenberg without content
    book = fetch_book(book_id, include_content=False)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book

@app.post("/books/{book_id}/generate-embeddings")
async def generate_book_embeddings(book_id: str):
    """Generate embeddings for a book's content."""
    try:
        print(f"\n=== Starting embedding generation for book {book_id} ===")
        
        # Fetch the content
        content = fetch_book_content(book_id)
        if not content:
            print(f"No content found for book {book_id}")
            raise HTTPException(status_code=404, detail="Book content not found")
        
        # Compile full content from pages
        full_content = "\n\n".join(content['pages'])
        
        # Process the content (chunk and generate embeddings)
        processed_content = process_text(full_content, book_id)
        
        print(f"✅ Embedding generation completed for book {book_id}")
        print(f"Chunks: {processed_content['statistics']['total_chunks']}")
        
        return {
            "status": "success",
            "message": "Embeddings generated successfully",
            "statistics": processed_content['statistics']
        }
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating embeddings: {str(e)}"
        )

@app.get("/books/{book_id}/content")
async def get_book_content(book_id: str):
    print(f"\n=== Fetching content for book {book_id} ===")
    
    # Fetch the content (already paginated)
    content = fetch_book_content(book_id)
    if not content:
        print("No content found")
        raise HTTPException(status_code=404, detail="Book content not found")
    
    print(f"✅ Content fetched successfully")
    print(f"Pages: {content['total_pages']}")
    
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
        
        # Load existing embeddings
        chunk_embeddings = load_embeddings(book_id)
        
        # If no embeddings found, generate them
        if not chunk_embeddings:
            print(f"No embeddings found for book {book_id}, generating them now...")
            try:
                # Fetch the content
                content = fetch_book_content(book_id)
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
        
        # For now, we'll use all chunks as context
        # (Later we can implement semantic search to find relevant chunks)
        context = "\n\n".join(chunks[:3])
        
        # Generate response using LLM
        response = await llm_handler.generate_response(
            query=request.query,
            context=context,
            temperature=request.temperature
        )
        
        return {
            "query": request.query,
            "response": response,
            "book_id": book_id
        }
        
    except Exception as e:
        print(f"❌ Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        ) 
   