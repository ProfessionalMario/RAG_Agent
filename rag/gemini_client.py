import google.generativeai as genai
from core.config import GEMINI_FILES
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in environment")

model = genai.GenerativeModel("gemini-1.5-flash")


def call_gemini(column_data, report_text):
    try:
        prompt = f"""
You are a data governance assistant.

Column:
{column_data}

Dataset Summary:
{report_text[:8000]}

Return STRICT JSON:
{{
  "decision": "ALLOW | RESTRICT | UNKNOWN",
  "reason": "...",
  "confidence": 0.0-1.0
}}
"""

        # Attach uploaded files (RAG-style)
        files = [genai.get_file(f) for f in GEMINI_FILES]

        response = model.generate_content(
            contents=files + [prompt]
        )

        return response.text or ""

    except Exception as e:
        return ""