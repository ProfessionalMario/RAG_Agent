"""
analyzer.py

----------------------------------------
Project Structure Analyzer for RAG_Agent
----------------------------------------

Purpose:
--------
This module recursively analyzes the RAG_Agent project folder, producing
a clean, filtered view of the project structure. It can:

1. Generate an ASCII tree of the folder structure.
2. Exclude specified folders, file types, or keywords.
3. Optionally be extended to extract Python functions, classes, and docstrings.

Exclusions:
-----------
- Folder-level exclusions via `EXCLUDE_PATHS`.
- Keyword-based exclusions via `EXCLUDE_KEYWORDS`.
- File-type exclusions via `EXCLUDE_EXTENSIONS`.
- Hidden/system files are automatically skipped.

Usage:
------
1. Place this script inside `total_context/`.
2. Configure your exclusions at the top of the script:

    EXCLUDE_PATHS = {"models/minilm", "data/pdfs"}
    EXCLUDE_KEYWORDS = {"onnx", "openvino"}
    EXCLUDE_EXTENSIONS = {".pdf"}

3. Run the script from the `RAG_Agent` folder:

    python total_context/analyzer.py

4. The generated ASCII tree will be written to:

    total_context/project_tree.txt

5. Optionally, you can adjust `EXCLUDE_PATHS`, `EXCLUDE_KEYWORDS`, or
   `EXCLUDE_EXTENSIONS` dynamically for different runs.

Example Output (ASCII tree):
----------------------------
RAG_Agent
├── core
│   ├── config.py

Notes:
------
- This tool is primarily for documentation and for generating
  LLM-friendly project maps.
- For LLM ingestion, consider extending this script to export
  a JSON structure with docstrings, functions, and classes.
"""

import os
print("Running from:", os.getcwd())
print("Script location:", os.path.abspath(__file__))
IGNORE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".vscode",
    ".ipynb_checkpoints"
}

IGNORE_FILES = {
    ".DS_Store",
}

IGNORE_EXTENSIONS = {
    ".pyc",
    ".pyo"
}

# 🔥 NEW: manual exclusions (relative to project root)
EXCLUDE_PATHS = {
    "models/minilm",     # 👈 your case
    # "data/pdfs",
}

# 🔥 OPTIONAL: keyword-based nuking
EXCLUDE_KEYWORDS = {
    # "onnx",
    # "openvino",
}


def should_exclude(full_path: str, root_path: str) -> bool:
    rel_path = os.path.relpath(full_path, root_path).replace("\\", "/")

    # Exact path match
    for excluded in EXCLUDE_PATHS:
        if rel_path.startswith(excluded):
            return True

    # Keyword match
    for keyword in EXCLUDE_KEYWORDS:
        if keyword in rel_path:
            return True

    return False


def is_ignored(name: str, path: str, root_path: str) -> bool:
    if name.startswith("."):
        return True

    if should_exclude(path, root_path):
        return True

    if os.path.isdir(path) and name in IGNORE_DIRS:
        return True

    if name in IGNORE_FILES:
        return True

    if any(name.endswith(ext) for ext in IGNORE_EXTENSIONS):
        return True

    return False


def build_clean_tree(root_path: str, current_path: str = None, prefix: str = "") -> str:
    if current_path is None:
        current_path = root_path

    tree = ""

    try:
        entries = sorted(os.listdir(current_path))

        entries = [
            e for e in entries
            if not is_ignored(e, os.path.join(current_path, e), root_path)
        ]

        for i, entry in enumerate(entries):
            path = os.path.join(current_path, entry)
            connector = "└── " if i == len(entries) - 1 else "├── "

            tree += f"{prefix}{connector}{entry}\n"

            if os.path.isdir(path):
                extension = "    " if i == len(entries) - 1 else "│   "
                tree += build_clean_tree(root_path, path, prefix + extension)

    except Exception as e:
        tree += f"{prefix}⚠️ Error: {e}\n"

    return tree


def write_tree_to_file(output_path: str, content: str):
    dir_name = os.path.dirname(output_path)

    # ✅ Only create dir if it exists
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    # print(f"✅ Tree saved to: {output_path}")
    # print("File exists after write:", os.path.exists(output_path))


def main(project_dir: str = None, output_file: str = None):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = project_dir or os.path.dirname(BASE_DIR)
    OUTPUT_FILE = output_file or os.path.join(BASE_DIR, "project_tree.txt")

    tree = build_clean_tree(PROJECT_DIR)
    write_tree_to_file(OUTPUT_FILE, tree)
    print(f"✅ Project tree saved to {OUTPUT_FILE}")