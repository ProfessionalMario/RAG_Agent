from pathlib import Path
import shutil
import PyPDF2
from docling.document_converter import DocumentConverter
from core.logger import get_logger

logger = get_logger(__name__)


class PDFParser:
    def __init__(self, pdf_path: str, tmp_dir: str = "tmp_pdf_chunks", chunk_pages: int = 10):
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            logger.error(f"[PDF_PARSER] Invalid path: {pdf_path}")
            raise ValueError(f"Invalid PDF path: {pdf_path}")

        self.tmp_dir = Path(tmp_dir)
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir()

        self.chunk_pages = chunk_pages
        self.converter = DocumentConverter()
        self.chunk_files = []

        logger.info(f"[PDF_PARSER] Initialized for {self.pdf_path}")

    # -----------------------------
    # Split PDF safely
    # -----------------------------
    def _split_pdf(self):
        try:
            reader = PyPDF2.PdfReader(str(self.pdf_path))
            total_pages = len(reader.pages)

            self.chunk_files.clear()

            for start in range(0, total_pages, self.chunk_pages):
                end = min(start + self.chunk_pages, total_pages)

                writer = PyPDF2.PdfWriter()
                for i in range(start, end):
                    writer.add_page(reader.pages[i])

                chunk_path = self.tmp_dir / f"chunk_{start+1}_{end}.pdf"

                with open(chunk_path, "wb") as f:
                    writer.write(f)

                self.chunk_files.append(chunk_path)

            logger.info(f"[PDF_PARSER] Split into {len(self.chunk_files)} chunks")
            return total_pages

        except Exception as e:
            logger.exception("[PDF_PARSER] Failed during PDF splitting")
            return 0

    # -----------------------------
    # Extract markdown safely
    # -----------------------------
    def _extract_markdown(self):
        all_text = []
        failed_chunks = []

        for chunk_pdf in self.chunk_files:
            try:
                result = self.converter.convert(str(chunk_pdf))
                md = result.document.export_to_markdown()

                all_text.append(md)

            except Exception as e:
                logger.exception(f"[PDF_PARSER] Failed chunk: {chunk_pdf.name}")
                failed_chunks.append(str(chunk_pdf.name))

        logger.info(f"[PDF_PARSER] Extraction complete | failed={len(failed_chunks)}")
        return all_text, failed_chunks

    # -----------------------------
    # PUBLIC API
    # -----------------------------
    def parse(self) -> dict:
        try:
            total_pages = self._split_pdf()
            md_blocks, failed_chunks = self._extract_markdown()

            logger.info("[PDF_PARSER] Parse completed successfully")

            return {
                "markdown_blocks": md_blocks,
                "meta": {
                    "source": str(self.pdf_path),
                    "total_pages": total_pages,
                    "chunk_pages": self.chunk_pages,
                    "num_chunks": len(self.chunk_files),
                    "failed_chunks": failed_chunks,
                },
            }

        except Exception:
            logger.exception("[PDF_PARSER] Critical failure in parse()")
            return {
                "markdown_blocks": [],
                "meta": {
                    "source": str(self.pdf_path),
                    "error": "parse_failed"
                }
            }

    # -----------------------------
    # Cleanup safe
    # -----------------------------
    def cleanup(self):
        try:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
            logger.info("[PDF_PARSER] Temp files cleaned")
        except Exception:
            logger.exception("[PDF_PARSER] Cleanup failed")


# -----------------------------
# TEST MODULE
# -----------------------------
if __name__ == "__main__":
    parser = PDFParser("data/pdfs/Data_Analysis.pdf", chunk_pages=5)

    result = parser.parse()

    logger.info(f"[TEST] Blocks: {len(result['markdown_blocks'])}")
    logger.info(f"[TEST] Pages: {result['meta'].get('total_pages')}")
    logger.info(f"[TEST] Failed: {result['meta'].get('failed_chunks')}")