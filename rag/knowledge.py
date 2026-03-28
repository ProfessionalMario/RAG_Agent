"""
Knowledge Base Loader – Developer Guide

This module provides utilities to manage and preprocess PDFs/TXT into
knowledge chunks for embedding, retrieval, or analysis.

--------------------------------------------------------------------------------
Callable Functions:
--------------------------------------------------------------------------------

has_knowledge() -> bool
    - Checks if knowledge chunks exist.
    - Output: True/False

load_chunks() -> List[str]
    - Loads saved knowledge chunks from storage.
    - Output: List of chunks (empty if not found or error).

get_latest_pdf_timestamp() -> float
    - Returns the most recent modification time of PDFs in the data folder.
    - Output: Timestamp as float.

ensure_knowledge_ready()
    - Checks if knowledge chunks are missing or outdated.
    - Rebuilds chunks if needed.

rebuild_knowledge()
    - Recreates knowledge chunks from all PDFs.
    - Calls `load_all_pdfs` and `save_chunks`.

save_chunks(chunks: List[str])
    - Saves a list of text chunks to storage.
    - Input: chunks
    - Output: Written JSON file.

load_all_pdfs(folder_path: str) -> List[str]
    - Loads and splits all PDFs in a folder into chunks.
    - Input: folder path
    - Output: List of text chunks

load_txt(path: str) -> List[str]
    - Loads a TXT file as lines, ignoring empties.
    - Input: path
    - Output: List of non-empty stripped lines

load_pdf(path: str) -> List[str]
    - Loads a PDF file and splits pages into chunks.
    - Input: path
    - Output: List of text chunks

split_text(text: str, chunk_size: int = 300) -> List[str]
    - Splits a long string into word-based chunks.
    - Input: text, optional chunk size
    - Output: List of text chunks

--------------------------------------------------------------------------------
Usage Notes:
--------------------------------------------------------------------------------
from knowledge import load_chunks, ensure_knowledge_ready

ensure_knowledge_ready()
chunks = load_chunks()
print(f"Loaded {len(chunks)} chunks")

"""

from typing import List
from pypdf import PdfReader
import os
import json
from core.logger import get_logger

logger = get_logger(__name__)

import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK_PATH = os.path.join(BASE_DIR, "storage", "knowledge_base", "eda_chunks.json")
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")


def has_knowledge():
    return os.path.exists(CHUNK_PATH)


def load_chunks():
    """
    Load knowledge chunks (no path required externally)
    """

    try:
        if not os.path.exists(CHUNK_PATH):
            logger.warning("Chunks not found")
            return []

        with open(CHUNK_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        logger.info(f"Loaded {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logger.exception(f"Failed to load chunks: {str(e)}")
        return []

def get_latest_pdf_timestamp():
    times = []
    for f in os.listdir(PDF_DIR):
        if f.endswith(".pdf"):
            times.append(os.path.getmtime(os.path.join(PDF_DIR, f)))
    return max(times) if times else 0


def ensure_knowledge_ready():
    if not os.path.exists(CHUNK_PATH):
        logger.info("No chunks → building knowledge")
        rebuild_knowledge()
        return

    chunk_time = os.path.getmtime(CHUNK_PATH)
    pdf_time = get_latest_pdf_timestamp()

    if pdf_time > chunk_time:
        logger.info("PDF changed → rebuilding knowledge")
        rebuild_knowledge()


def rebuild_knowledge():
    chunks = load_all_pdfs(PDF_DIR)
    save_chunks(chunks)


def save_chunks(chunks):
    try:
        os.makedirs(os.path.dirname(CHUNK_PATH), exist_ok=True)

        with open(CHUNK_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        logger.info(f"Saved {len(chunks)} chunks")

    except Exception as e:
        logger.error(f"Failed saving chunks: {str(e)}")

    

def load_all_pdfs(folder_path: str) -> List[str]:
    """
    Load and chunk all PDFs in a folder
    """
    all_chunks = []

    try:
        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                path = os.path.join(folder_path, file)
                chunks = load_pdf(path)
                all_chunks.extend(chunks)

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

    except Exception as e:
        logger.error(f"Failed loading PDFs: {str(e)}")
        return []

def load_txt(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        logger.error(f"TXT load failed: {path}")
        return []


def load_pdf(path: str) -> List[str]:
    try:
        reader = PdfReader(path)

        chunks = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                chunks.extend(split_text(text))

        logger.info(f"Loaded PDF: {path}")
        return chunks

    except Exception as e:
        logger.error(f"PDF load failed: {path}")
        return []


def split_text(text: str, chunk_size: int = 300) -> List[str]:
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


if __name__ == "__main__":

    pdf_folder = "data/pdfs"
    output_path = "storage/knowledge_base/eda_chunks.json"

    chunks = load_all_pdfs(pdf_folder)
    save_chunks(chunks, output_path)

    print("✅ Knowledge base created")