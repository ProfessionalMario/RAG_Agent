"""
Docstring Extractor
-------------------
Extracts all human-readable module, class, and function docstrings from a Python project.

Usage:
    python -m total_context.docstring_extractor <project_path>

Example:
    python -m total_context.docstring_extractor X:\RAG_Agent\RAG_Agent

Features:
- Ignores irrelevant folders/files (configurable)
- Skips __init__.py unless it has a docstring
- Prints summary to terminal
- Saves output in docstrings.txt in the script folder
"""

import os
import ast
import sys

# -----------------------------
# Configuration
# -----------------------------
IGNORE_DIRS = {"__pycache__", ".ipynb_checkpoints", "logs", "models", "data"}
IGNORE_FILES = {".DS_Store"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "docstrings.txt")

# -----------------------------
# Extract docstrings from a single Python file
# -----------------------------
def extract_docstrings(file_path):
    result = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=file_path)

        module_doc = ast.get_docstring(tree)
        if module_doc:
            result.append(f"Module: {os.path.relpath(file_path, SCRIPT_DIR)}")
            result.append(f"  Module Docstring: {module_doc}\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_doc = ast.get_docstring(node)
                if class_doc:
                    result.append(f"  Class: {node.name} (Doc: yes)")
                    for func in [n for n in node.body if isinstance(n, ast.FunctionDef)]:
                        func_doc = ast.get_docstring(func)
                        result.append(f"    Function: {func.name} (Doc: {'yes' if func_doc else 'no'})")
                    result.append("")
            elif isinstance(node, ast.FunctionDef):
                # Skip functions inside classes (already captured)
                if isinstance(getattr(node, 'parent', None), ast.ClassDef):
                    continue
                func_doc = ast.get_docstring(node)
                if func_doc:
                    result.append(f"  Function: {node.name} (Doc: yes)")
        return result
    except Exception as e:
        print(f"⚠️ Failed to parse {file_path}: {e}")
        return []

# -----------------------------
# Walk project folder
# -----------------------------
def extract_project_docstrings(project_path):
    docstrings = []
    for root, dirs, files in os.walk(project_path):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            if file.endswith(".py") and file not in IGNORE_FILES:
                # Skip __init__.py if no docstring
                file_path = os.path.join(root, file)
                if file == "__init__.py" and not ast.get_docstring(ast.parse(open(file_path, "r", encoding="utf-8").read())):
                    continue

                # Assign parents to AST nodes for proper class function detection
                with open(file_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        for child in ast.iter_child_nodes(node):
                            setattr(child, "parent", node)

                docstrings.extend(extract_docstrings(file_path))
    return docstrings

# -----------------------------
# Main
# -----------------------------
def main():
    project_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "..")

    if not os.path.exists(project_path):
        print(f"❌ Project path does not exist: {project_path}")
        return

    print(f"Scanning project at: {project_path}\n")
    docstrings = extract_project_docstrings(project_path)

    if not docstrings:
        print("⚠️ No docstrings found.")
        return

    summary = "\n".join(docstrings)
    # print(summary[:2000] + "\n...")  # Print first 2000 chars to avoid flooding

    # Save to file
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\n✅ Docstring extraction complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 