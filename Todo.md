Reports-> Answer with knowledge -> generate useful decisions via decision agents that can later be converted to actions. 


RAW INPUT (Paragraphs / PDF Extracted Text)
        │
        ▼
[load_text()]
        │
        ▼
[normalize_text()]
  - fix encoding (� → proper chars)
  - unify whitespace
        │
        ▼
[remove_metadata_lines()]
  - © lines
  - headers/footers
        │
        ▼
[remove_index_like_lines()]
  - "CONTENTS", page numbers, dotted lines
        │
        ▼
[remove_front_matter()]
  - preface / TOC blocks (if detected)
        │
        ▼
[fix_broken_words()]   🆕
  - "Sim ilarly" → "Similarly"
  - "measur ing" → "measuring"
  - "hyphen\nation" → "hyphenation"
        │
        ▼
[fix_spacing_errors()] 🆕
  - "oof" → "of"
  - "nuancees" → "nuances"
  - remove duplicate chars
        │
        ▼
[handle_math_and_symbols()] 🆕
  - preserve inline math (E = mc^2)
  - preserve LaTeX blocks
  - remove noisy symbol separators
        │
        ▼
[remove_noise_patterns()] 🆕
  - footnotes (*, [1,2])
  - OCR artifacts
  - random separators (---, $$ if not needed)
        │
        ▼
[reconstruct_paragraphs()] 🆕
  - merge broken lines
  - fix sentence continuity
        │
        ▼
[validate_clean_text()]
  - if empty → FAIL FAST
        │
        ▼
[chunk_text()]   (ONLY ONCE)
  - semantic / fixed-size chunks
        │
        ▼
[optional_llm_cleanup()] 🆕 (ONLY IF NEEDED)
  - final polish on chunks
  - fix edge-case OCR errors
        │
        ▼
FINAL CLEAN CHUNKS → EMBEDDING → VECTOR STORE








Things to mention in resume, 
it not only uses docling for pdf parsing but also cleans them internally using wordninja ensuring the retrieved content is not a garbage. 
Also its aware of the context in two way the entire project code is aware of its own code and it will be easier to understand the structure for llms on what the code does, also it builds context based on the report so that it answers contextually. 