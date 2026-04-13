import re
import logging
from typing import List

# -----------------------------
# LOGGER SETUP
# -----------------------------
logger = logging.getLogger("chunker")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -----------------------------
# CORE CHUNKING LOGIC
# -----------------------------

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # Keep track of the 'original_start' to ensure we always move forward
        original_start = start

        # 1. Handle the START "Snap"
        if start > 0:
            next_space = text.find(" ", start)
            # Only snap if the space is within a reasonable distance (the overlap)
            if next_space != -1 and next_space < (start + chunk_overlap):
                start = next_space + 1

        # 2. Determine the END "Snap"
        end = start + chunk_size
        if end < text_len:
            last_space = text.rfind(" ", start, end)
            if last_space != -1 and last_space > start:
                end = last_space
        
        # 3. Extract
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 4. Slide the window 
        # Crucial Fix: Ensure start ALWAYS advances beyond original_start
        new_start = end - chunk_overlap
        start = max(new_start, original_start + 1)

    return chunks

# --- SURGICAL INSPECTION TOOL ---
def debug_retrieval_chunks(chunks: List[str]):
    """Call this in main to verify you aren't storing garbage."""
    if not chunks:
        print("DEBUG: No chunks generated.")
        return

    print(f"\n{'='*20} CHUNK QUALITY CHECK {'='*20}")
    print(f"Total Chunks: {len(chunks)}")
    
    # Check a sample (First and Middle)
    indices = [0, len(chunks) // 2] if len(chunks) > 1 else [0]
    
    for i in indices:
        print(f"\n[CHUNK #{i}]")
        print(f"CHARS: {len(chunks[i])}")
        # Show first 100 and last 100 chars to check for "cut-offs"
        print(f"START: {chunks[i][:100]}")
        print(f"END:   {chunks[i][-100:]}")
        print("-" * 50)
    print(f"{'='*60}\n")

# -----------------------------
# MODULE INTERFACE
# -----------------------------

def generate_extraction_chunks(text: str, size: int = 512, overlap: int = 64) -> List[str]:
    try:
        if not text or len(text.strip()) == 0:
            return []

        clean_text = re.sub(r'\s+', ' ', text).strip()
        logger.info(f"[CHUNKER] Segmenting {len(clean_text)} chars into clean-cut chunks.")
        
        chunks = chunk_text(clean_text, size, overlap)
        
        logger.info(f"[CHUNKER] Created {len(chunks)} clean chunks.")
        return chunks

    except Exception as e:
        logger.exception(f"[CHUNKER] Failure: {e}")
        return []

# -----------------------------
# TEST MODE
# -----------------------------

def test():
    sample_text = (
        "Machine learning is a subset of AI. It focuses on learning from data. "
        "1. Introduction to models. Models can be supervised or unsupervised. "
        "This is an important concept. DATA SCIENCE OVERVIEW. "
        "Feature engineering improves performance."
    )

    print("\n--- RUNNING CLEAN-CUT CHUNKER TEST ---")
    # Size 60, Overlap 20
    chunks = generate_extraction_chunks(sample_text, size=60, overlap=20)
    
    for i, c in enumerate(chunks):
        s_count = c.count('.') + c.count('?') + c.count('!')
        print(f"{{'chunk_id': {i}, 'text': '{c}', 'size': {len(c)}, 'sentence_count': {s_count}}}")

if __name__ == "__main__":
    test()