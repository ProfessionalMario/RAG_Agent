# # import pymupdf4llm

# # # Get clean markdown text for a PDF
# # # markdown = pymupdf4llm.to_markdown("input.pdf")

# # # Or get plain text
# # plain = pymupdf4llm.to_text("data\pdfs\Data_Analysis.pdf")

# # # Save to file
# # from pathlib import Path
# # Path("output.txt").write_text(plain, encoding="utf-8")

# from pathlib import Path
# import shutil
# import PyPDF2
# from docling.document_converter import DocumentConverter
# # from rag.retriever import chunk_text,is_valid

# class PDFParser:
#     def __init__(
#         self,
#         pdf_path: str,
#         tmp_dir: str = "tmp_pdf_chunks",
#         chunk_pages: int = 10,
#     ):
#         self.pdf_path = Path(pdf_path)

#         if not self.pdf_path.exists():
#             raise ValueError(f"[ERROR] Invalid PDF path: {pdf_path}")

#         self.tmp_dir = Path(tmp_dir)
#         self.tmp_dir.mkdir(exist_ok=True)

#         self.chunk_pages = chunk_pages
#         self.converter = DocumentConverter()
#         self.chunk_files: list[Path] = []

#     # -----------------------------
#     # Split PDF into smaller chunks
#     # -----------------------------
#     def _split_pdf(self):
#         reader = PyPDF2.PdfReader(str(self.pdf_path))
#         total_pages = len(reader.pages)

#         self.chunk_files.clear()

#         for start in range(0, total_pages, self.chunk_pages):
#             end = min(start + self.chunk_pages, total_pages)

#             writer = PyPDF2.PdfWriter()
#             for i in range(start, end):
#                 writer.add_page(reader.pages[i])

#             chunk_path = self.tmp_dir / f"chunk_{start+1}_{end}.pdf"

#             with open(chunk_path, "wb") as f:
#                 writer.write(f)

#             self.chunk_files.append(chunk_path)

#         return total_pages

#     # -----------------------------
#     # Convert each chunk → markdown
#     # -----------------------------
#     # def _extract_markdown(self):
#     #     all_chunks = []
#     #     failed_chunks = []

#     #     for chunk_pdf in self.chunk_files:
#     #         try:
#     #             result = self.converter.convert(str(chunk_pdf))
#     #             doc = result.document

#     #             md = doc.export_to_markdown()

#     #             # ✅ chunk immediately
#     #             chunks = chunk_text(md)

#     #             # ✅ filter valid chunks
#     #             valid_chunks = [c for c in chunks if is_valid(c)]

#     #             all_chunks.extend(valid_chunks)

#     #         except Exception as e:
#     #             print(f"[WARN] Failed: {chunk_pdf.name} → {e}")
#     #             failed_chunks.append(chunk_pdf.name)

#     #     return all_chunks, failed_chunks
#     def _extract_markdown(self):
#         all_blocks = [] # Change name to blocks, not chunks
#         failed_chunks = []

#         for chunk_pdf in self.chunk_files:
#             try:
#                 result = self.converter.convert(str(chunk_pdf))
#                 md = result.document.export_to_markdown()
                
#                 # We NO LONGER chunk here. We just collect the raw MD.
#                 all_blocks.append(md)

#             except Exception as e:
#                 print(f"[WARN] Failed: {chunk_pdf.name} → {e}")
#                 failed_chunks.append(chunk_pdf.name)

#         return all_blocks, failed_chunks
#     # -----------------------------
#     # Public API → Parse PDF
#     # -----------------------------
#     def parse(self) -> dict:
#         total_pages = self._split_pdf()
        
#         # This now returns raw MD blocks
#         md_blocks, failed_chunks = self._extract_markdown()

#         return {
#             "chunks": md_blocks,  # We call them chunks here so 'knowledge.py' doesn't crash
#             "meta": {
#                 "source": str(self.pdf_path),
#                 "total_pages": total_pages,
#                 "failed_chunks": failed_chunks,
#             },
#         }

#     # # -----------------------------
#     # # Optional: Save markdown
#     # # -----------------------------
#     # def save_markdown(self, output_path: str = "output.md") -> str:
#     #     result = self.parse()

#     #     out_path = Path(output_path)
#     #     out_path.write_text(result["raw_md"], encoding="utf-8")

#     #     print(f"[INFO] Markdown saved → {out_path}")
#     #     return str(out_path)

#     # -----------------------------
#     # Cleanup temp files
#     # -----------------------------
#     def cleanup(self):
#         shutil.rmtree(self.tmp_dir, ignore_errors=True)


#     # def save_temp_md(self, md_path: str = "output.md"):
#     #     result = self.parse()

#     #     md_text = result["raw_md"]

#     #     # Path(md_path).write_text(md_text, encoding="utf-8")

#     #     # print(f"[INFO] Saved markdown → {md_path}")
#     #     if "num_chunks" in result:
#     #        print(f"[INFO] Processed chunks → {result['num_chunks']}")
        
#     #     # Auto-clean temp markdown
#     #     Path(md_path).unlink(missing_ok=True)
#     #     return result