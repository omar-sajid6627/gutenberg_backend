from nltk.tokenize import sent_tokenize
import nltk
import os
# Set environment variable for tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from typing import List, Dict, Any
import numpy as np
import pickle
from pathlib import Path
import time

# Create a directory for storing embeddings if it doesn't exist
EMBEDDINGS_DIR = Path("backend/app/data/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Download required NLTK data (you only need to do this once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab')

# Lazy initialization of embedding model to prevent issues with forking
embeddings_model = None

def get_embeddings_model():
    """
    Get the embeddings model, initializing it if necessary.
    This lazy initialization helps avoid issues with multiprocessing.
    
    Returns:
        FastEmbedEmbeddings: The embedding model
    """
    global embeddings_model
    if embeddings_model is None:
        print("Initializing embedding model...")
        embeddings_model = FastEmbedEmbeddings()
    return embeddings_model

def generate_embeddings(chunks: List[str]) -> List[np.ndarray]:
    """
    Generate embeddings for a list of text chunks.
    
    Args:
        chunks (List[str]): List of text chunks to generate embeddings for
    
    Returns:
        List[np.ndarray]: List of embedding vectors
    """
    try:
        vectors = []
        total_chunks = len(chunks)
        
        print(f"\n=== Generating embeddings for {total_chunks} chunks ===")
        
        # Get embedding model (will initialize if needed)
        model = get_embeddings_model()
        
        # Limit how many chunks we actually embed for better performance
        chunks_to_process = chunks
        if total_chunks > 50:
            print(f"Processing a sample of chunks for better performance...")
            # Process first few, middle few, and last few
            first = 20
            last = 20
            middle_start = (total_chunks - 10) // 2 
            chunks_to_process = chunks[:first] + chunks[middle_start:middle_start+10] + chunks[-last:]
            print(f"Processing {len(chunks_to_process)} chunks instead of {total_chunks}")
            
        # Start a timer to track how long this takes
        start_time = time.time()
        
        for i, chunk in enumerate(chunks_to_process):
            print(f"Processing chunk {i + 1}/{len(chunks_to_process)}")
            embedding = model.embed_query(chunk)
            vectors.append(embedding)
            
            # Print progress every 10 chunks
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                print(f"Progress: {i + 1}/{len(chunks_to_process)} chunks processed in {elapsed:.1f}s ({avg_time:.2f}s per chunk)")
            
        print(f"✅ Successfully generated {len(vectors)} embeddings in {time.time() - start_time:.1f} seconds")
        print(f"Embedding dimension: {len(vectors[0])}")
        
        return vectors
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {str(e)}")
        raise Exception(f"Failed to generate embeddings: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks based on sentence boundaries.
    
    Args:
        text (str): The input text to be chunked
        chunk_size (int): Maximum size of each chunk in words (default: 1024)
        overlap (int): Number of words to overlap between chunks (default: 100)
    
    Returns:
        list[str]: List of text chunks
    """
    if not text:
        return []

    try:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            if current_length + sentence_length > chunk_size:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Calculate how many sentences to keep for overlap
                overlap_words = 0
                overlap_sentences = []
                
                for sent in reversed(current_chunk):
                    sent_words = len(sent.split())
                    if overlap_words + sent_words <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_words += sent_words
                    else:
                        break
                
                # Keep overlap sentences for next chunk
                current_chunk = overlap_sentences
                current_length = overlap_words

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add the last remaining chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Limit total chunks for performance if there are too many
        max_chunks = 50  # Set a maximum number of chunks to process
        if len(chunks) > max_chunks:
            print(f"⚠️ Too many chunks ({len(chunks)}), sampling down to {max_chunks}...")
            # Take evenly distributed samples across the text
            step = len(chunks) // max_chunks
            sampled_chunks = [chunks[i * step] for i in range(max_chunks - 1)]
            # Always include the last chunk for context completion
            sampled_chunks.append(chunks[-1])
            chunks = sampled_chunks

        print(f"✅ Text successfully chunked into {len(chunks)} parts")
        print(f"Average chunk size: {sum(len(chunk.split()) for chunk in chunks) / len(chunks):.0f} words")
        
        return chunks

    except Exception as e:
        print(f"❌ Error during text chunking: {str(e)}")
        raise Exception(f"Failed to chunk text: {str(e)}")

def get_chunk_info(chunks: list[str]) -> dict:
    """
    Get information about the chunks for debugging purposes.
    
    Args:
        chunks (list[str]): List of text chunks
    
    Returns:
        dict: Dictionary containing chunk statistics
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "total_words": 0,
            "average_chunk_size": 0,
            "chunk_sizes": []
        }

    chunk_sizes = [len(chunk.split()) for chunk in chunks]
    
    return {
        "total_chunks": len(chunks),
        "total_words": sum(chunk_sizes),
        "average_chunk_size": sum(chunk_sizes) / len(chunks),
        "chunk_sizes": chunk_sizes
    }

def save_embeddings(book_id: str, chunks: List[str], vectors: List[np.ndarray]) -> str:
    """
    Save chunks and their embeddings to a pickle file.
    
    Args:
        book_id (str): ID of the book
        chunks (List[str]): List of text chunks
        vectors (List[np.ndarray]): List of embedding vectors
    
    Returns:
        str: Path to the saved pickle file
    """
    try:
        # Create a list of dictionaries containing chunks and their embeddings
        chunk_embeddings = [
            {
                "text": chunk,
                "embedding": embedding
            }
            for chunk, embedding in zip(chunks, vectors)
        ]
        
        # Create filename using book_id
        filename = EMBEDDINGS_DIR / f"book_{book_id}_embeddings.pkl"
        
        # Save to pickle file
        with open(filename, "wb") as f:
            pickle.dump(chunk_embeddings, f)
        
        print(f"✅ Embeddings saved successfully to {filename}")
        return str(filename)
        
    except Exception as e:
        print(f"❌ Error saving embeddings: {str(e)}")
        raise Exception(f"Failed to save embeddings: {str(e)}")

def load_embeddings(book_id: str) -> List[Dict[str, Any]]:
    """
    Load chunks and their embeddings from a pickle file.
    
    Args:
        book_id (str): ID of the book
    
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing chunks and their embeddings
    """
    try:
        filename = EMBEDDINGS_DIR / f"book_{book_id}_embeddings.pkl"
        
        if not filename.exists():
            print(f"❌ No embeddings file found for book {book_id}")
            return None
        
        with open(filename, "rb") as f:
            chunk_embeddings = pickle.load(f)
        
        print(f"✅ Embeddings loaded successfully from {filename}")
        print(f"Total chunks loaded: {len(chunk_embeddings)}")
        
        return chunk_embeddings
        
    except Exception as e:
        print(f"❌ Error loading embeddings: {str(e)}")
        raise Exception(f"Failed to load embeddings: {str(e)}")

def process_text(text: str, book_id: str, chunk_size: int = 1024, overlap: int = 100) -> dict:
    """
    Process text by chunking it and generating embeddings, then save to pickle file.
    
    Args:
        text (str): Input text to process
        book_id (str): ID of the book
        chunk_size (int): Maximum size of each chunk in words
        overlap (int): Number of words to overlap between chunks
    
    Returns:
        dict: Dictionary containing chunks, embeddings, and statistics
    """
    try:
        # First, check if embeddings already exist
        existing_embeddings = load_embeddings(book_id)
        if existing_embeddings:
            print(f"Found existing embeddings for book {book_id}")
            chunks = [item["text"] for item in existing_embeddings]
            vectors = [item["embedding"] for item in existing_embeddings]
            
        else:
            # Limit the amount of text to process if it's extremely long
            if len(text) > 500000:  # About 500KB
                print(f"⚠️ Text is very long ({len(text)} chars), sampling sections...")
                first_third = text[:150000]
                middle_start = len(text) // 2 - 75000
                middle_third = text[middle_start:middle_start + 150000]
                last_third = text[-150000:]
                text = first_third + middle_third + last_third
                print(f"Reduced text from {len(text)} to {len(first_third) + len(middle_third) + len(last_third)} characters")
            
            print(f"Generating new embeddings for book {book_id}")
            # Generate chunks
            chunks = chunk_text(text, chunk_size, overlap)
            # Generate embeddings
            vectors = generate_embeddings(chunks)
            # Save embeddings
            save_embeddings(book_id, chunks, vectors)
        
        chunk_statistics = get_chunk_info(chunks)
        
        return {
            "chunks": chunks,
            "embeddings": vectors,
            "statistics": {
                **chunk_statistics,
                "embedding_dimension": len(vectors[0]) if vectors else 0
            },
            "source": "cache" if existing_embeddings else "new"
        }
        
    except Exception as e:
        print(f"❌ Error processing text: {str(e)}")
        raise Exception(f"Failed to process text: {str(e)}") 