import re
import logging
from typing import List
import wordninja

# -----------------------------
# LOGGER SETUP
# -----------------------------
logger = logging.getLogger("sentence_rebuilder")
# logging.basicConfig(level=logging.INFO)


# -----------------------------
# CORE CLEANING FUNCTIONS
# -----------------------------

def fix_hyphen_breaks(text: str) -> str:
    """
    preproces-\nsing OR preproces- sing → preprocessing
    """
    return re.sub(r"(\w+)-\s*\n?\s*(\w+)", r"\1\2", text)


def fix_missing_spaces(text: str) -> str:
    """
    dataQuality → data Quality
    """
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", text)


def fix_sentence_spacing(text: str) -> str:
    """
    .something → . Something
    """
    return re.sub(r"\.\s*([a-z])", lambda m: ". " + m.group(1).upper(), text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# -----------------------------
# INTELLIGENT WORD RECOVERY
# -----------------------------

def intelligent_word_recovery(text: str) -> str:
    """
    Replaces fix_glued_words and aggressive_fallback_split.
    Handles trailing punctuation and validates split quality to 
    prevent damaging good words.
    """
    tokens = text.split()
    final_tokens = []

    for tok in tokens:
        # 1. PEEL PUNCTUATION
        # Separates word from trailing chars (e.g., "imputation." -> "imputation", ".")
        match = re.match(r"^([\w-]+)([^\w]*)$", tok)
        if not match:
            final_tokens.append(tok)
            continue
            
        word, punct = match.groups()

        # 2. TRIGGER GATE
        # Only process words long enough to likely be 'glued' junk (e.g. > 10 chars)
        if len(word) > 10:
            # Wordninja works best on lowercase for frequency matching
            split_attempt = wordninja.split(word.lower())

            # 3. SAFETY VETO (The "NotebookLM" secret sauce)
            if len(split_attempt) > 1:
                # Calculate average fragment length
                # If it's too low (e.g., 'i n f r a'), it's a bad split; we skip it.
                avg_len = len(word) / len(split_attempt)
                
                if avg_len >= 3.0: 
                    # Re-capitalize first part if original word was capitalized
                    if word[0].isupper():
                        split_attempt[0] = split_attempt[0].capitalize()
                    
                    # Reattach the peeled punctuation to the very last fragment
                    split_attempt[-1] = split_attempt[-1] + punct
                    final_tokens.extend(split_attempt)
                    continue

        # Fallback: Keep original token if split was risky or unnecessary
        final_tokens.append(tok)

    return " ".join(final_tokens)


# -----------------------------
# MAIN REBUILDER
# -----------------------------

def rebuild_sentences(text: str) -> str:
    try:
        logger.info("[REBUILDER] Starting reconstruction")

        # Step 1: Structural Repair
        text = fix_hyphen_breaks(text)
        text = fix_missing_spaces(text)

        # Step 2: Semantic Repair (The heavy lifter)
        text = intelligent_word_recovery(text)

        # Step 3: Aesthetic/Grammar Repair
        text = fix_sentence_spacing(text)
        text = normalize_whitespace(text)

        logger.info("[REBUILDER] Reconstruction complete")
        return text

    except Exception as e:
        logger.exception(f"[REBUILDER] Failed: {e}")
        return text


# -----------------------------
# TEST RUN
# -----------------------------

def test():
    sample = """
    datapreprocessingMissingvaluesinnumericalcolumns
    canbehandledusingmeanormedianimputation.
    preproces-
    sing techniques improve dataQuality.
    """

    print("\n--- INPUT ---")
    print(sample)

    output = rebuild_sentences(sample)

    print("\n--- OUTPUT ---")
    print(output)


if __name__ == "__main__":
    test()