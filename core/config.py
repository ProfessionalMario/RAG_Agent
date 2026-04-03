USE_GEMINI = True
MAX_CONTEXT_CHARS = 12000
ENGINE = "rag" , "gemini" 
# options: "rag", "gemini"

import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in environment")

GEMINI_FILES = [
    "files/isi0ez78fwz8",
    "files/rj4y5qdjkkzi",
    "files/kfjjqjql3bn4"
]


