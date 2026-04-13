import re
from typing import List
from core.logger import get_logger

logger = get_logger("text_cleaner")

# text_cleaner.py
import re
from typing import List
from core.logger import get_logger

logger = get_logger("text_cleaner")


# def light_clean(md: str) -> str:
#     import re
#     md = re.sub(r'(\w)\s+(\w)', r'\1\2', md)
#     return md

def remove_noise_symbols(text: str) -> str:
    import re

    # Remove weird replacement characters
    text = re.sub(r"[�]+", " ", text)

    cleaned_lines = []

    for line in text.split("\n"):
        line = line.strip()

        # Handle table-like lines
        if "|" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]

            # Keep meaningful text parts only (avoid pure numbers)
            parts = [p for p in parts if not p.isdigit()]

            if parts:
                cleaned_lines.append(" ".join(parts))
        else:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Normalize spacing
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()

def normalize(text: str) -> str:
    if not text:
        return ""
    try:
        # 1. Direct replacement (safely removes the 'garbage' symbols)
        text = text.replace("\ufffd", " ") 
        
        # 2. Clean whitespace without complex regex sets
        return " ".join(text.split())

    except Exception as e:
        # Professional one-liner (No more Traceback flood)
        print(f"[!] Clean Fail: {str(e)[:50]}")
        return text

def remove_tables(text: str) -> str:
    lines = text.split("\n")
    out = []
    for line in lines:
        line = line.strip()
        if not line or set(line.replace("|", "").strip()) in [{"-"}, {"="}, set()]:
            continue
        if "|" in line:
            parts = [p.strip() for p in line.split("|") if len(p.strip()) > 1]
            if len(parts) >= 2: out.append(" ".join(parts))
            continue
        out.append(line)
    return " ".join(out)

def is_valid_chunk(text: str) -> bool:
    # Increased minimum length to ensure quality for BGE-Small
    if not text or len(text) < 50: return False
    if len(text.split()) < 8: return False
    return True




# # -----------------------------
# # MAIN
# # -----------------------------
# if __name__ == "__main__":
#     try:
#         logger.info("[CLEANER] standalone test started")
        
#         logger.info("[CLEANER] completed")

#     except Exception as e:
#         logger.exception(f"[CLEANER] crash: {e}")