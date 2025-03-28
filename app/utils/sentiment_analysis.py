import re
import nltk
import json
import time
from pathlib import Path
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from typing import Dict, List, Any, Optional

# Create necessary directories
SENTIMENT_DIR = Path("data/sentiment")
SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache for sentiment analysis results
_sentiment_cache = {}
# Cache expiration in seconds (12 hours)
SENTIMENT_CACHE_EXPIRATION = 43200

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('sentiment/vader_lexicon')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('wordnet', quiet=True)

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()
clean_pattern = re.compile(r'[^a-z\s]')

def get_cached_sentiment(book_id: str) -> Optional[Dict[str, Any]]:
    """
    Get sentiment analysis from cache if available and not expired.
    
    Args:
        book_id (str): ID of the book
        
    Returns:
        Optional[Dict[str, Any]]: Cached sentiment analysis or None if not found/expired
    """
    # Check memory cache first
    if book_id in _sentiment_cache:
        cache_entry = _sentiment_cache[book_id]
        # Check if cache is still valid
        if time.time() - cache_entry["timestamp"] < SENTIMENT_CACHE_EXPIRATION:
            print(f"✅ Using cached sentiment analysis for book {book_id}")
            return cache_entry["result"]
        else:
            # Expired, remove from memory cache
            del _sentiment_cache[book_id]
    
    # Check disk cache
    cache_file = SENTIMENT_DIR / f"book_{book_id}_sentiment.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
            
            # Check if disk cache is still valid
            if time.time() - cache_data.get("timestamp", 0) < SENTIMENT_CACHE_EXPIRATION:
                # Update memory cache and return result
                _sentiment_cache[book_id] = {
                    "result": cache_data["result"],
                    "timestamp": cache_data["timestamp"]
                }
                print(f"✅ Using disk cached sentiment analysis for book {book_id}")
                return cache_data["result"]
        except Exception as e:
            print(f"❌ Error reading sentiment cache file: {str(e)}")
    
    # Not in cache or expired
    return None

def save_sentiment_to_cache(book_id: str, result: Dict[str, Any]) -> None:
    """
    Save sentiment analysis to both memory and disk cache.
    
    Args:
        book_id (str): ID of the book
        result (Dict[str, Any]): Sentiment analysis results to cache
    """
    timestamp = time.time()
    
    # Save to memory cache
    _sentiment_cache[book_id] = {
        "result": result,
        "timestamp": timestamp
    }
    
    # Save to disk cache
    try:
        cache_file = SENTIMENT_DIR / f"book_{book_id}_sentiment.json"
        with open(cache_file, "w") as f:
            json.dump({
                "result": result,
                "timestamp": timestamp
            }, f)
        print(f"✅ Saved sentiment analysis to cache for book {book_id}")
    except Exception as e:
        print(f"❌ Error saving sentiment to cache file: {str(e)}")

def preprocess_text(content: str) -> List[str]:
    """
    Cleans and preprocesses text by removing special characters, stopwords, and applying lemmatization.
    
    Args:
        content (str): The raw text content to preprocess
        
    Returns:
        List[str]: A list of preprocessed tokens
    """
    # Convert to lowercase and remove special characters
    content = clean_pattern.sub('', content.lower())

    # Tokenize and remove stopwords
    words = [word for word in word_tokenize(content) if word not in stop_words]

    # Apply lemmatization and filter short words
    processed_words = [lemmatizer.lemmatize(word) for word in words if len(word) > 2]
    
    return processed_words

def analyze_sentiment(content: str) -> Dict[str, float]:
    """
    Performs sentiment analysis sentence by sentence and returns average scores.
    
    Args:
        content (str): The text to analyze
        
    Returns:
        Dict[str, float]: Average sentiment scores
    """
    # Skip sentiment analysis if text is too short
    if len(content) < 10:
        return {"neg": 0, "neu": 1.0, "pos": 0, "compound": 0}
    
    # Split text into sentences
    sentences = sent_tokenize(content)
    
    # Use just a sample of sentences if there are too many (for performance)
    if len(sentences) > 1000:
        # Take an evenly distributed sample throughout the text
        sample_size = 1000
        step = len(sentences) // sample_size
        sentences = sentences[::step][:sample_size]
    
    # Analyze each sentence
    sentiment_scores = [analyzer.polarity_scores(sentence) for sentence in sentences]
    
    # Aggregate scores (average of each component)
    avg_scores = {
        "neg": sum(score["neg"] for score in sentiment_scores) / len(sentiment_scores),
        "neu": sum(score["neu"] for score in sentiment_scores) / len(sentiment_scores),
        "pos": sum(score["pos"] for score in sentiment_scores) / len(sentiment_scores),
        "compound": sum(score["compound"] for score in sentiment_scores) / len(sentiment_scores)
    }
    
    return avg_scores

def get_word_frequencies(tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Get word frequencies for wordcloud generation on the frontend.
    
    Args:
        tokens (List[str]): Preprocessed tokens
        
    Returns:
        List[Dict[str, Any]]: List of words and their counts for frontend wordcloud
    """
    # Get word frequencies
    word_counts = Counter(tokens)
    
    # Sort by frequency (descending)
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Limit to top 150 words for frontend performance
    top_words = sorted_words[:150]
    
    # Format for frontend
    return [{"text": word, "value": count} for word, count in top_words]

def analyze_book_sentiment(content: str, book_id: str) -> Dict[str, Any]:
    """
    Performs sentiment analysis of book content and extracts words for wordcloud.
    
    Args:
        content (str): The book content as text
        book_id (str): ID of the book
        
    Returns:
        Dict[str, Any]: Analysis results including sentiment and word frequencies
    """
    # Check cache first
    cached_result = get_cached_sentiment(book_id)
    if cached_result:
        return cached_result
    
    try:
        print(f"Starting sentiment analysis for book {book_id}...")
        
        # Preprocess tokens for wordcloud
        print("Preprocessing text for wordcloud...")
        tokens = preprocess_text(content)
        
        # Get word frequencies for wordcloud
        print("Extracting word frequencies...")
        word_data = get_word_frequencies(tokens)
        
        # Analyze sentiment on original content
        print("Analyzing sentiment...")
        sentiment_scores = analyze_sentiment(content)
        
        # Determine overall sentiment
        overall = "positive"
        if sentiment_scores["compound"] < -0.05:
            overall = "negative"
        elif sentiment_scores["compound"] <= 0.05:
            overall = "neutral"
        
        result = {
            "book_id": book_id,
            "sentiment": {
                "positive": sentiment_scores["pos"],
                "negative": sentiment_scores["neg"],
                "neutral": sentiment_scores["neu"],
                "compound": sentiment_scores["compound"],
                "overall": overall
            },
            "overall_sentiment": overall,
            "wordcloud_data": {
                "words": word_data,
                "word_count": len(word_data),
                "total_words_analyzed": len(tokens)
            }
        }
        
        print(f"✅ Completed sentiment analysis for book {book_id}")
        
        # Save to cache
        save_sentiment_to_cache(book_id, result)
        
        return result
        
    except Exception as e:
        print(f"❌ Error analyzing book sentiment: {str(e)}")
        raise Exception(f"Failed to analyze book sentiment: {str(e)}")
