

# import os
# import re
# import faiss
# import pickle
# import numpy as np
# from nltk.tokenize import sent_tokenize
# from sentence_transformers import SentenceTransformer
# from pathlib import Path
# from extractor.pdf_parsing import PDFParser  # your module
# import nltk
# from core.logger import get_logger
# from core.exceptions import ParserError
# from typing import List
# import os
# import pickle
# from typing import List
# from rag.retriever import FAISS_DIR, DOCS_PATH, INDEX_PATH
# from rag.retriever import FaissRetriever, get_model
# from extractor.sentence_rebuilder import rebuild_sentences
# from extractor.text_cleaner import normalize, remove_tables, is_valid_chunk
# from extractor.chunker import generate_extraction_chunks,debug_retrieval_chunks
# logger = get_logger(__name__)
# # -----------------------------
# # CONFIG
# # -----------------------------
# PDF_PATH = "data/pdfs/scikit.pdf"
# TMP_MD_PATH = "output2.md"
# STORAGE_DIR = Path("storage")
# INDEX_PATH = STORAGE_DIR / "faiss.index"
# # META_PATH = STORAGE_DIR / "meta.pkl"
# CHUNK_SIZE = 150
# CHUNK_OVERLAP = 30
# # MODEL_NAME = "all-MiniLM-L6-v2"
# MODEL_NAME = "BAAI/bge-small-en-v1.5"

# STORAGE_DIR.mkdir(exist_ok=True)
# nltk.download('punkt')

# import hashlib

# # -----------------------------
# # 1️⃣ Load saved knowledge chunks
# # -----------------------------
# def load_chunks() -> List[str]:
#     if not os.path.exists(DOCS_PATH):
#         return []

#     # 🚨 Critical fix: check empty file
#     if os.path.getsize(DOCS_PATH) == 0:
#         logger.warning("[LOAD CHUNKS] Empty file detected. Deleting...")
#         os.remove(DOCS_PATH)
#         return []

#     try:
#         with open(DOCS_PATH, "rb") as f:
#             data = pickle.load(f)
#             logger.info(f"[VERIFY] DOCS_PATH size: {os.path.getsize(DOCS_PATH)} bytes")
#             if isinstance(data, list):
#                 return data
#             elif isinstance(data, dict) and "chunks" in data:
#                 return data["chunks"]
#             else:
#                 return []

#     except Exception as e:
#         logger.warning(f"[LOAD CHUNKS FAILED] {e}")

#         # 🚨 Auto-recovery
#         os.remove(DOCS_PATH)
#         logger.warning("[LOAD CHUNKS] Corrupted file deleted")

#         return []

# # -----------------------------
# # 2️⃣ Ensure knowledge is ready
# # -----------------------------
# def ensure_knowledge_ready():
#     """
#     Ensure FAISS index + chunks exist.
#     If missing → build from PDF pipeline.
#     """

#     index_exists = os.path.exists(INDEX_PATH)
#     docs_exists = os.path.exists(DOCS_PATH)

#     if index_exists and docs_exists and load_chunks():
#         logger.info("[KNOWLEDGE] FAISS + chunks already exist. Skipping rebuild.")
#         return

#     logger.warning("[KNOWLEDGE] Missing index or docs. Building knowledge base...")

#     # 🚀 Call your real pipeline
#     main()


# def compute_hash(text: str, config: dict) -> str:
#     combined = text + str(config)
#     return hashlib.sha256(combined.encode("utf-8")).hexdigest()

# import hashlib

# CONFIG = {
#     "model": MODEL_NAME,
#     "chunk_size": CHUNK_SIZE,
#     "overlap": CHUNK_OVERLAP,
#     "clean_version": "v1.0"
# }

# HASH_PATH = "storage/index.hash"

# def save_hash(hash_val: str):
#     with open(HASH_PATH, "w") as f:
#         f.write(hash_val)

# def load_hash():
#     import os
#     if not os.path.exists(HASH_PATH):
#         return None
#     return open(HASH_PATH).read().strip()

# # ==============================
# # RAG CORE PIPELINE
# # ==============================

# def build_faiss_index(chunks: list, model_name: str = MODEL_NAME):
#     model = SentenceTransformer(model_name)
#     embeddings = model.encode(chunks, show_progress_bar=True)
#     embeddings = np.array(embeddings).astype("float32")
#     dim = embeddings.shape[1]

#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)
#     # os.makedirs(STORAGE_DIR, exist_ok=True)
#     logger.info(f"[FAISS BUILD] Total chunks: {len(chunks)}")
#     logger.debug(f"[FAISS SAMPLE] {chunks[:2]}")
#     STORAGE_DIR.mkdir(exist_ok=True)
#     faiss.write_index(index, str(INDEX_PATH))
#     with open(DOCS_PATH, "wb") as f:
#         pickle.dump(chunks, f)

#     print(f"[INFO] FAISS index saved at {INDEX_PATH}")
#     return index

# # ==============================
# # FULL PIPELINE CALL
# # ==============================
# def peek_chunks(chunks, sample_size=3):
#     """
#     Diagnostic tool to verify chunk quality without memory overhead.
#     """
#     if not chunks:
#         print("\n[!] TEST FAILED: No chunks found.")
#         return

#     print(f"\n--- SURGICAL CHUNK DIAGNOSTIC (Total: {len(chunks)}) ---")
    
#     # Check first, middle, and last to ensure consistency
#     indices = [0, len(chunks) // 2, len(chunks) - 1]
    
#     for i in indices:
#         if i < len(chunks):
#             print(f"\n[CHUNK #{i}]")
#             print(f"Length: {len(chunks[i])} chars")
#             # Print a snippet of the chunk
#             print(f"Content: {chunks[i][:300]}...") 
#             print("-" * 30)

# def inspect_pipeline_quality(chunks, num_samples=3):
#     """
#     Surgically inspects chunks to ensure cleaning and windowing logic 
#     is producing high-quality retrieval candidates.
#     """
#     if not chunks:
#         print("\n[!] CRITICAL: No chunks found. Pipeline is broken.")
#         return

#     total = len(chunks)
#     # Pick start, middle, and end indices to see variety
#     sample_indices = [0, total // 2, total - 1] if total > 2 else range(total)

#     print(f"\n{'='*60}")
#     print(f"📊 PIPELINE INSPECTION: {total} Total Chunks Generated")
#     print(f"{'='*60}")

#     for idx in sample_indices:
#         chunk = chunks[idx]
#         print(f"\n[SAMPLE CHUNK #{idx}]")
#         print(f"📏 Size: {len(chunk)} characters")
#         print(f"📝 Preview: {chunk[:400]}...") # First 400 chars
#         print(f"{'-'*30}")
    
#     print("\n[VERIFICATION CHECKLIST]")
#     print("1. Glued words? (e.g. 'theprocess') -> If yes, wordninja failed.")
#     print("2. Tables? (e.g. '|---|') -> If yes, remove_tables failed.")
#     print("3. Cutoffs? -> Check if sentences end abruptly or flow logically.")
#     print(f"{'='*60}\n")


# def main():
#     # 1. LIGHTWEIGHT HASH CHECK (Zero Memory Footprint)
#     # Get file stats (size and last modified time) to create a quick fingerprint
#     file_stats = os.stat(PDF_PATH)
#     current_fingerprint = f"{PDF_PATH}_{file_stats.st_size}_{file_stats.st_mtime}"
    
#     stored_hash = load_hash() # Adjust load_hash to handle this string or its hash

#     if current_fingerprint == stored_hash:
#         print("[INFO] Source file unchanged. Skipping heavy processing.")
#         return

#     # 2. Extraction & Refining (The memory-heavy part)
#     parser = PDFParser(pdf_path=PDF_PATH, chunk_pages=10)
#     parsed_data = parser.parse() 
    
#     processed_chunks = []
#     for block in parsed_data["markdown_blocks"]:
#         fixed_text = rebuild_sentences(block)
#         clean_text = normalize(remove_tables(fixed_text))
#         chunks = generate_extraction_chunks(clean_text, size=512, overlap=64)
        
#         valid_chunks = [c for c in chunks if is_valid_chunk(c)]
#         processed_chunks.extend(valid_chunks)

#     # print("===================","Quality check block","=======================")
#     # peek_chunks(processed_chunks)
#     # # CLEAR BLOCK FROM MEMORY
#     # inspect_pipeline_quality(processed_chunks)
#     # debug_retrieval_chunks(processed_chunks)
#     # print("==========================================")
#     # del block 
#     # 3. FAISS Build
#     if processed_chunks:
#         print(f"[INFO] Building index for {len(processed_chunks)} chunks...")
#         build_faiss_index(processed_chunks)
        
#         save_hash(current_fingerprint)
#         print("[INFO] Success.")
        
#         # Explicitly clear chunks before finishing
#         del processed_chunks
#     else:
#         print("[ERROR] No chunks found.")

# if __name__ == "__main__":
#     main()



























import os
import pickle
import hashlib
import faiss
import numpy as np
import nltk
from pathlib import Path
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from core.logger import get_logger
from extractor.sentence_rebuilder import rebuild_sentences
from extractor.text_cleaner import normalize, remove_tables, is_valid_chunk
from extractor.chunker import generate_extraction_chunks

# Initialize
logger = get_logger(__name__)
nltk.download('punkt', quiet=True)

# -----------------------------
# CONFIG
# -----------------------------
MD_SOURCE_PATH = "data/pdfs/grounded_brain.md"
STORAGE_DIR = Path("storage")
INDEX_PATH = STORAGE_DIR / "faiss.index"
DOCS_PATH = STORAGE_DIR / "docs.pkl"
HASH_PATH = STORAGE_DIR / "index.hash"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

STORAGE_DIR.mkdir(exist_ok=True)

# -----------------------------
# 1️⃣ Diagnostic Tools
# -----------------------------

def peek_chunks(chunks: List[str]):
    """Diagnostic tool to verify chunk quality without memory overhead."""
    if not chunks:
        logger.error("[DIAGNOSTIC] FAILED: No chunks found.")
        return

    print(f"\n--- SURGICAL CHUNK DIAGNOSTIC (Total: {len(chunks)}) ---")
    # Check first, middle, and last to ensure consistency
    indices = [0, len(chunks) // 2, len(chunks) - 1]
    
    for i in indices:
        if i < len(chunks):
            print(f"\n[CHUNK #{i}]")
            print(f"Length: {len(chunks[i])} chars")
            print(f"Content: {chunks[i][:300]}...") 
            print("-" * 30)


def load_chunks() -> List[str]:
    if not os.path.exists(DOCS_PATH):
        return []

    # 🚨 Critical fix: check empty file
    if os.path.getsize(DOCS_PATH) == 0:
        logger.warning("[LOAD CHUNKS] Empty file detected. Deleting...")
        os.remove(DOCS_PATH)
        return []

    try:
        with open(DOCS_PATH, "rb") as f:
            data = pickle.load(f)
            logger.info(f"[VERIFY] DOCS_PATH size: {os.path.getsize(DOCS_PATH)} bytes")
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "chunks" in data:
                return data["chunks"]
            else:
                return []

    except Exception as e:
        logger.warning(f"[LOAD CHUNKS FAILED] {e}")

        # 🚨 Auto-recovery
        os.remove(DOCS_PATH)
        logger.warning("[LOAD CHUNKS] Corrupted file deleted")

        return []



def inspect_pipeline_quality(chunks: List[str]):
    """Surgically inspects chunks for cleaning and windowing logic artifacts."""
    if not chunks:
        print("\n[!] CRITICAL: No chunks found. Pipeline is broken.")
        return

    total = len(chunks)
    sample_indices = [0, total // 2, total - 1] if total > 2 else range(total)

    print(f"\n{'='*60}")
    print(f"📊 PIPELINE INSPECTION: {total} Total Chunks Generated")
    print(f"{'='*60}")

    for idx in sample_indices:
        chunk = chunks[idx]
        print(f"\n[SAMPLE CHUNK #{idx}]")
        print(f"📏 Size: {len(chunk)} characters")
        print(f"📝 Preview: {chunk[:400]}...") 
        print(f"{'-'*30}")
    
    print("\n[VERIFICATION CHECKLIST]")
    print("1. Glued words? -> Check rebuild_sentences.")
    print("2. Tables? -> Check remove_tables.")
    print("3. Cutoffs? -> Check chunker overlap.")
    print(f"{'='*60}\n")

# -----------------------------
# 2️⃣ Hashing & State Management
# -----------------------------

def compute_fingerprint(file_path: str) -> str:
    """Creates a unique hash based on file stats and content."""
    try:
        stats = os.stat(file_path)
        combined = f"{file_path}_{stats.st_size}_{stats.st_mtime}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
    except Exception as e:
        logger.error(f"[HASHING] Failed to compute fingerprint: {e}")
        return ""

def is_knowledge_synced() -> bool:
    """Checks if the local FAISS index matches the current source file."""
    if not HASH_PATH.exists() or not INDEX_PATH.exists() or not DOCS_PATH.exists():
        return False
    
    current_hash = compute_fingerprint(MD_SOURCE_PATH)
    stored_hash = HASH_PATH.read_text().strip()
    return current_hash == stored_hash

# -----------------------------
# 3️⃣ Build & Load
# -----------------------------

def build_knowledge_base():
    """Main pipeline: MD -> Clean -> Chunk -> Embed -> FAISS."""
    try:
        logger.info(f"🚀 Starting Knowledge Build from {MD_SOURCE_PATH}")
        
        # 1. Read Source
        raw_text = Path(MD_SOURCE_PATH).read_text(encoding="utf-8")
        
        # 2. Process & Refine
        # We split by headers to maintain some contextual local boundaries
        blocks = raw_text.split("\n## ")
        processed_chunks = []
        
        for block in blocks:
            # Apply your fine-tuned cleaning stack
            fixed_text = rebuild_sentences(block)
            clean_text = normalize(remove_tables(fixed_text))
            
            # Generate overlapping windows
            chunks = generate_extraction_chunks(clean_text, size=512, overlap=64)
            valid_chunks = [c for c in chunks if is_valid_chunk(c)]
            processed_chunks.extend(valid_chunks)

        if not processed_chunks:
            logger.error("[BUILD] No valid chunks generated. Check source file.")
            return

        # 3. Embedding & Indexing
        logger.info(f"[BUILD] Embedding {len(processed_chunks)} chunks using {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode(processed_chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        # 4. Save Artifacts
        faiss.write_index(index, str(INDEX_PATH))
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(processed_chunks, f)
        
        HASH_PATH.write_text(compute_fingerprint(MD_SOURCE_PATH))
        
        logger.info("✅ Knowledge base built and saved successfully.")
        
        # Diagnostics
        peek_chunks(processed_chunks)
        inspect_pipeline_quality(processed_chunks)

    except Exception as e:
        logger.error(f"❌ Critical Failure during Knowledge Build: {e}")
        raise

def ensure_knowledge_ready():
    """External API to ensure index exists and is up to date."""
    if is_knowledge_synced():
        logger.info("[KNOWLEDGE] FAISS index is up-to-date. Ready for retrieval.")
        return
    
    logger.warning("[KNOWLEDGE] Stale or missing index. Rebuilding...")
    build_knowledge_base()

# -----------------------------
# 4️⃣ Test Block
# -----------------------------
if __name__ == "__main__":
    try:
        print("--- Knowledge Module Test Mode ---")
        ensure_knowledge_ready()
        
        # Verify Load
        with open(DOCS_PATH, "rb") as f:
            chunks = pickle.load(f)
            print(f"[TEST] Successfully loaded {len(chunks)} chunks.")
            
    except Exception as e:
        print(f"[TEST] Failed: {e}")