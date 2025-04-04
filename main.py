from fastapi import FastAPI, HTTPException, BackgroundTasks
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
from app.utils.sentiment_analysis import analyze_book_sentiment, get_cached_sentiment
from app.utils.content_cache import clear_cache
import numpy as np
from mangum import Mangum  # Required for Vercel
import uvicorn
import time
import nltk

# Set NLTK data path to a writable location in Koyeb
nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

# Download required NLTK data at startup
try:
    nltk.download('punkt', quiet=True, download_dir=nltk_data_dir)
    nltk.download('punkt_tab', quiet=True, download_dir=nltk_data_dir)
    nltk.download('stopwords', quiet=True, download_dir=nltk_data_dir)
    nltk.download('vader_lexicon', quiet=True, download_dir=nltk_data_dir)
    nltk.download('averaged_perceptron_tagger', quiet=True, download_dir=nltk_data_dir)
    nltk.download('maxent_ne_chunker', quiet=True, download_dir=nltk_data_dir)
    nltk.download('words', quiet=True, download_dir=nltk_data_dir)
    nltk.download('wordnet', quiet=True, download_dir=nltk_data_dir)
    print("✅ NLTK resources downloaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Failed to download NLTK resources: {str(e)}")

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
                
                print(f"Generated new embeddings for book {book_id}")
            except Exception as e:
                print(f"Error generating embeddings: {str(e)}")
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
        print(f"Error processing query: {str(e)}")
        # Print the full error traceback for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/books/{book_id}/sentiment", response_model=Dict[str, Any])
@app.post("/books/{book_id}/sentiment", response_model=Dict[str, Any])
async def analyze_book_sentiment_endpoint(book_id: str, background_tasks: BackgroundTasks):
    """
    Get sentiment analysis for a book.
    Supports both GET and POST methods for flexibility.
    """
    try:
        print(f"\n=== Analyzing sentiment for book {book_id} ===")
        
        # First check if we already have a cached sentiment analysis
        cached_result = get_cached_sentiment(book_id)
        if cached_result:
            return cached_result
        
        # Get book content
        content_data = get_book_content(book_id)
        
        if not content_data:
            raise HTTPException(status_code=404, detail=f"Book content for ID {book_id} not found")
        
        # Extract the full content from pages
        if "pages" not in content_data or not content_data["pages"]:
            raise HTTPException(status_code=404, detail=f"No page content available for book {book_id}")
        
        # Combine all pages into one text
        full_content = "\n\n".join(content_data["pages"])
        content_length = len(full_content)
        print(f"Content retrieved: {content_length:,} characters")
        
        # For extremely large books, provide a warning and sample just beginning and end
        if content_length > 1_500_000:
            print(f"⚠️ Extremely large book detected: {content_length:,} characters!")
            print("Sampling content instead of full analysis to prevent timeout")
            # Take first 400K, middle 200K and last 400K chars
            first_part = full_content[:400000]
            middle_start = (len(full_content) - 200000) // 2
            middle_part = full_content[middle_start:middle_start + 200000]
            last_part = full_content[-400000:]
            full_content = first_part + "...\n\n[CONTENT SAMPLED FOR PERFORMANCE]...\n\n" + middle_part + "...\n\n[CONTENT SAMPLED FOR PERFORMANCE]...\n\n" + last_part
            print(f"Reduced to {len(full_content):,} characters for analysis")
        
        # Create a placeholder result to return immediately
        placeholder_result = {
            "book_id": book_id,
            "sentiment": {
                "positive": 0,
                "negative": 0,
                "neutral": 1.0,
                "compound": 0,
                "overall": "neutral"
            },
            "overall_sentiment": "neutral",
            "wordcloud_data": {
                "words": [],
                "word_count": 0,
                "total_words_analyzed": 0
            },
            "status": "processing",
            "message": "Sentiment analysis is being processed in the background. Please check again in a few moments."
        }
        
        # Add the actual analysis to background tasks
        def run_sentiment_analysis():
            try:
                start_time = time.time()
                print("Starting background sentiment analysis...")
                result = analyze_book_sentiment(full_content, book_id)
                print(f"✅ Background sentiment analysis completed in {time.time() - start_time:.2f} seconds")
                return result
            except Exception as e:
                print(f"❌ Error in background sentiment analysis: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
        
        # Add the analysis task to background tasks
        background_tasks.add_task(run_sentiment_analysis)
        
        # Return the placeholder result immediately
        return placeholder_result
        
    except HTTPException as he:
        # Re-raise HTTP exceptions as-is
        raise he
    except Exception as e:
        print(f"❌ Error in sentiment endpoint: {str(e)}")
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
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