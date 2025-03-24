from typing import Dict, Any, Optional
from ..scraper import fetch_book_content
from .content_cache import get_cached_content, save_content_to_cache

def get_book_content(book_id: str) -> Optional[Dict[str, Any]]:
    """
    Get book content from cache or fetch it if not cached.
    
    Args:
        book_id (str): ID of the book
        
    Returns:
        Optional[Dict[str, Any]]: Book content or None if not found
    """
    # Try to get from cache first
    cached_content = get_cached_content(book_id)
    if cached_content:
        return cached_content
    
    # Not in cache, fetch from source
    print(f"\n=== Fetching content for book {book_id} ===")
    content = fetch_book_content(book_id)
    
    if content:
        print(f"✅ Content fetched successfully")
        print(f"Pages: {content['total_pages']}")
        
        # Save to cache for future use
        save_content_to_cache(book_id, content)
        
    else:
        print(f"❌ No content found for book {book_id}")
    
    return content
