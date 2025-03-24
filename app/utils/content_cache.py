from typing import Dict, Any, Optional
import time
from pathlib import Path
import json
import os

# Create cache directory
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache
_memory_cache = {}
# Cache expiration in seconds (default: 1 hour)
CACHE_EXPIRATION = 3600

def get_cached_content(book_id: str) -> Optional[Dict[str, Any]]:
    """
    Get book content from cache if available and not expired.
    
    Args:
        book_id (str): ID of the book
        
    Returns:
        Optional[Dict[str, Any]]: Cached content or None if not found/expired
    """
    # Check memory cache first
    if book_id in _memory_cache:
        cache_entry = _memory_cache[book_id]
        # Check if cache is still valid
        if time.time() - cache_entry["timestamp"] < CACHE_EXPIRATION:
            print(f"✅ Using in-memory cached content for book {book_id}")
            return cache_entry["content"]
        else:
            # Expired, remove from memory cache
            del _memory_cache[book_id]
    
    # Check disk cache
    cache_file = CACHE_DIR / f"book_{book_id}_content.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
            
            # Check if disk cache is still valid
            if time.time() - cache_data.get("timestamp", 0) < CACHE_EXPIRATION:
                # Update memory cache and return content
                _memory_cache[book_id] = {
                    "content": cache_data["content"],
                    "timestamp": cache_data["timestamp"]
                }
                print(f"✅ Using disk cached content for book {book_id}")
                return cache_data["content"]
        except Exception as e:
            print(f"❌ Error reading cache file: {str(e)}")
    
    # Not in cache or expired
    return None

def save_content_to_cache(book_id: str, content: Dict[str, Any]) -> None:
    """
    Save book content to both memory and disk cache.
    
    Args:
        book_id (str): ID of the book
        content (Dict[str, Any]): Book content to cache
    """
    timestamp = time.time()
    
    # Save to memory cache
    _memory_cache[book_id] = {
        "content": content,
        "timestamp": timestamp
    }
    
    # Save to disk cache
    try:
        cache_file = CACHE_DIR / f"book_{book_id}_content.json"
        with open(cache_file, "w") as f:
            json.dump({
                "content": content,
                "timestamp": timestamp
            }, f)
        print(f"✅ Saved content to cache for book {book_id}")
    except Exception as e:
        print(f"❌ Error saving to cache file: {str(e)}")

def clear_cache(book_id: Optional[str] = None) -> None:
    """
    Clear cache for a specific book or all books.
    
    Args:
        book_id (Optional[str]): ID of the book to clear cache for, or None to clear all
    """
    if book_id:
        # Clear specific book from memory cache
        if book_id in _memory_cache:
            del _memory_cache[book_id]
        
        # Clear specific book from disk cache
        cache_file = CACHE_DIR / f"book_{book_id}_content.json"
        if cache_file.exists():
            os.remove(cache_file)
        print(f"✅ Cleared cache for book {book_id}")
    else:
        # Clear all memory cache
        _memory_cache.clear()
        
        # Clear all disk cache
        for cache_file in CACHE_DIR.glob("book_*_content.json"):
            os.remove(cache_file)
        print("✅ Cleared all book content cache") 