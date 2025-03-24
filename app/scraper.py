import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, Optional
from .models import Book
import textwrap

def fetch_and_format_book(book_id, words_per_page=500):
    """
    Fetches and formats a book from Project Gutenberg.

    Args:
        book_id (int): The ID of the book.
        words_per_page (int): Number of words per page for pagination.

    Returns:
        list: A list of pages, each containing a chunk of text.
    """
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    response = requests.get(url)

    if response.status_code != 200:
        return ["Failed to fetch book content"]

    raw_text = response.text
    print(raw_text)

    # Remove Gutenberg metadata (header and footer)
    start_match = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG", raw_text)
    end_match = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG", raw_text)

    if start_match and end_match:
        raw_text = raw_text[start_match.end():end_match.start()].strip()

    # Normalize spaces and remove unwanted special characters
    raw_text = re.sub(r"\s+", " ", raw_text)  # Replace multiple spaces/newlines with a single space
    raw_text = raw_text.replace("_", "")  # Remove underscores used for italicization

    print("PreProcessed raw_text: ", raw_text)
    # Split into paragraphs based on double newlines
    paragraphs = raw_text.split("\n\n")

    # Format each paragraph for better readability
    formatted_paragraphs = [textwrap.fill(p.strip(), width=80) for p in paragraphs if p.strip()]

    # Join paragraphs with double newlines for spacing
    formatted_text = "\n\n".join(formatted_paragraphs)

    # Paginate the text
    words = formatted_text.split()
    pages = [" ".join(words[i:i+words_per_page]) for i in range(0, len(words), words_per_page)]

    return pages

def fetch_book_metadata(book_id: str) -> Optional[str]:
    """Fetch book metadata from Project Gutenberg."""
    metadata_url = f"https://www.gutenberg.org/ebooks/{book_id}"
    try:
        response = requests.get(metadata_url)
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        return None

def scrape_gutenberg_book(metadata: str) -> Optional[Dict]:
    """Parse book metadata from HTML content."""
    soup = BeautifulSoup(metadata, 'html.parser')

    # Extract Title
    title = soup.find("td", itemprop="headline")
    title_text = title.get_text(strip=True) if title else "N/A"

    # Extract author
    author_tag = soup.find('a', rel='marcrel:aut')
    author = author_tag.text.strip() if author_tag else "Author not found"

    # Extract cover image URL
    cover_image = soup.find('img', {'class': 'cover-art'})
    cover_url = None
    if cover_image:
        img_src = cover_image['src']
        print(img_src)
        # Check if URL already includes the domain
        if img_src.startswith('http'):
            cover_url = img_src
            print(cover_url)

        else:
            cover_url = "https://www.gutenberg.org" + img_src

    # Extract Summary
    summary_container = soup.find(class_="summary-text-container")
    summary_text = summary_container.get_text(strip=True) if summary_container else "N/A"

    # Extract Notes
    notes = []
    print("Looking for notes in HTML structure...")
    
    note_elements = soup.find_all("th", string="Note")
    print(f"Found {len(note_elements)} note elements")
    
    for note in note_elements:
        note_text = note.find_next_sibling("td").get_text(strip=True)
        notes.append(note_text)
        print(f"Found note: {note_text}")

    print(f"All notes collected: {notes}")

    # Extract Subjects
    subjects = []
    for subject in soup.find_all("th", string="Subject"):
        subject_text = subject.find_next_sibling("td").get_text(strip=True)
        subjects.append(subject_text)

    # Extract downloads
    downloads_tag = soup.find("td", itemprop="interactionCount")
    downloads = 0
    if downloads_tag:
        match = re.search(r"(\d+)", downloads_tag.text)
        downloads = int(match.group(1)) if match else 0

    # Extract language
    language = None
    language_td = soup.find("tr", property="dcterms:language")
    if language_td:
        language = language_td.find("td").text.strip()

    # Extract release date
    release_date = None
    release_td = soup.find("tr", property="dcterms:issued")
    if release_td:
        release_date = release_td.find("td").text.strip()

    # Extract reading ease score
    reading_ease_score = None
    for note in notes:
        match = re.search(r"Reading ease score:\s*([\d.]+)", note)
        if match:
            reading_ease_score = float(match.group(1))
            break  # Only break if we found a match

    print(f"Final reading ease score being returned: {reading_ease_score}")
    return {
        "title": title_text,
        "author": author,
        "cover_url": cover_url,
        "summary": summary_text,
        "notes": notes,
        "subjects": subjects,
        "downloads": downloads,
        "language": language or "N/A",
        "release_date": release_date or "N/A",
        "reading_ease_score": reading_ease_score
    }

def fetch_book_content(book_id: str) -> Optional[Dict]:
    """Fetch book content from Project Gutenberg with pagination."""
    print(f"\n=== Fetching book content ===")
    print(f"Attempting to fetch content for book: {book_id}")
    try:
        pages = fetch_and_format_book(book_id)
        if not pages or pages[0] == "Failed to fetch book content":
            print("Failed to fetch book content")
            return None
            
        print(f"Successfully fetched {len(pages)} pages")
        return {
            "pages": pages,
            "total_pages": len(pages)
        }
    except Exception as e:
        print(f"Request failed with error: {str(e)}")
        return None

def fetch_book(book_id: str, include_content: bool = False) -> Optional[Book]:  
    """Fetch and parse a complete book from Project Gutenberg."""
    print(f"\n=== Starting fetch_book for book_id: {book_id} ===")
    
    print("Fetching book metadata...")
    metadata_html = fetch_book_metadata(book_id)
    if not metadata_html:
        print("Failed to fetch metadata")
        return None

    print("Scraping book metadata...")
    metadata = scrape_gutenberg_book(metadata_html)
    if not metadata:
        print("Failed to scrape metadata")
        return None

    # Only fetch content if requested
    content = None
    if include_content:
        print("Fetching book content...")
        content = fetch_book_content(book_id)
        if not content:
            print("Failed to fetch content")
            return None

    print("Creating Book object...")
    print(content)
    book = Book(
        id=book_id,
        title=metadata["title"],
        author=metadata["author"],
        language=metadata["language"],
        category=", ".join(metadata["subjects"]),
        ebook_no=book_id,
        release_date=metadata["release_date"],
        summary=metadata["summary"],
        cover_url=metadata["cover_url"],
        downloads=metadata["downloads"],
        reading_ease_score=metadata["reading_ease_score"],
        content=content
    )
    print("Book object created successfully")
    return book 