"""
Docstring Extractor – Clean Version
-----------------------------------

Scans a Python project folder and extracts all human-readable module, class,
and function docstrings into a JSON file.

Excludes:
- Empty modules (including __init__.py without docstring)
- Specified folders (models, data, etc.)
- Hidden/system folders or files
"""

from total_context.structure_extractor import build_clean_tree, write_tree_to_file
from total_context.docstring_extractor import extract_project_docstrings
import os
import ast
import json
from pathlib import Path

# ---------------------------
# CONFIG
# ---------------------------
EXCLUDE_PATHS = {"models", "data", ".pytest_cache", "__pycache__"}
EXCLUDE_KEYWORDS = {".git"}
EXCLUDE_EXTENSIONS = {".pdf", ".onnx", ".ot"}

# ---------------------------
# DOCSTRING EXTRACTION
# ---------------------------
def extract_docstrings(file_path: str):
    """Extract module, class, and function docstrings from a single Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception:
        return None

    module_doc = ast.get_docstring(tree)
    if not module_doc:
        return None  # skip empty modules

    doc_data = {"module_docstring": module_doc, "classes": {}, "functions": {}}

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cls_doc = ast.get_docstring(node) or ""
            doc_data["classes"][node.name] = {"docstring": cls_doc, "methods": {}}
            # methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    doc_data["classes"][node.name]["methods"][item.name] = ast.get_docstring(item) or ""

        elif isinstance(node, ast.FunctionDef):
            doc_data["functions"][node.name] = ast.get_docstring(node) or ""

    return doc_data

def extract_project_docstrings(project_dir: str):
    """Walk project folder and extract all human-readable docstrings."""
    project_dir = Path(project_dir).resolve()
    result = {}

    for root, dirs, files in os.walk(project_dir):
        # Filter out excluded folders
        dirs[:] = [d for d in dirs if d not in EXCLUDE_PATHS and all(k not in d for k in EXCLUDE_KEYWORDS)]

        for file in files:
            if file.endswith(".py") and not any(k in file for k in EXCLUDE_KEYWORDS):
                file_path = Path(root) / file
                doc = extract_docstrings(str(file_path))
                if doc:
                    relative_path = str(file_path.relative_to(project_dir))
                    result[relative_path] = doc

    return result


def main():
    # ---------------------------
    # Determine project paths
    # ---------------------------
    SCRIPT_DIR = Path(__file__).parent.resolve()
    PROJECT_DIR = SCRIPT_DIR.parent  # root of your project

    # ---------------------------
    # 1️⃣ Update project structure
    # ---------------------------
    structure_output = SCRIPT_DIR / "project_tree.txt"
    tree = build_clean_tree(PROJECT_DIR)
    write_tree_to_file(structure_output, tree)
    print(f"✅ Structure updated at {structure_output}")

    # ---------------------------
    # 2️⃣ Update docstrings
    # ---------------------------
    doc_output = SCRIPT_DIR / "project_doc.json"
    docs = extract_project_docstrings(PROJECT_DIR)
    with open(doc_output, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    print(f"✅ Docstrings updated at {doc_output}")

if __name__ == "__main__":
    main()