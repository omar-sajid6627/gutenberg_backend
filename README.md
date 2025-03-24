# Gutenberg API Backend

This is the backend service for the Gutenberg Frontend application. It provides APIs for accessing Project Gutenberg books.

## Requirements

- Python 3.11 or 3.12
- pip (Python package manager)

## Setup

1. Create a virtual environment with Python 3.11 or 3.12:
```bash
# If you have Python 3.11 installed:
python3.11 -m venv venv

# Or if you have Python 3.12 installed:
python3.12 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Upgrade pip to the latest version:
```bash
pip install --upgrade pip
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /`: Welcome message
- `GET /books`: List all books (paginated)
- `GET /books/{book_id}`: Get a specific book by ID
- `GET /books/search/{query}`: Search books by title, author, or summary
- `GET /health`: Health check endpoint

## API Documentation

Once the server is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Development

The project structure is:
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application and routes
│   ├── models.py        # Pydantic models
│   └── database.py      # Database operations
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Troubleshooting

If you encounter any issues during installation:

1. Make sure you're using Python 3.11 or 3.12
2. Try upgrading pip: `pip install --upgrade pip`
3. If you get build errors, you might need to install build tools:
   - On macOS: `xcode-select --install`
   - On Ubuntu/Debian: `sudo apt-get install python3-dev build-essential`
   - On Windows: Install Visual Studio Build Tools 