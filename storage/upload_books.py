
import google.generativeai as genai
from core.config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

file = genai.upload_file(path="data/pdfs/think stats.pdf")

print(file.name)

