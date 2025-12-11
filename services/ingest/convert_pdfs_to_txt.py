"""Convert PDF files in a directory to plain text files.

Usage:
  python services/ingest/convert_pdfs_to_txt.py --src-dir docs/Datasets/ChatGPT --out-dir services/ingest/raw_chatgpt_text

This is a small helper used before running the ingest pipeline when sources are PDFs.
"""

import argparse
from pathlib import Path

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


def extract_pdf_text(pdf_path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("PyPDF2 is not installed")
    text_parts = []
    try:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text_parts.append(txt)
    except Exception as e:
        print(f"Failed to read {pdf_path}: {e}")
    return "\n".join(text_parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pdf_files = [p for p in src.rglob("*.pdf")]
    if not pdf_files:
        print("No PDF files found in", src)
        return

    for pdf in pdf_files:
        txt = extract_pdf_text(pdf)
        if not txt.strip():
            print(f"No text extracted from {pdf}, skipping.")
            continue
        out_path = out / (pdf.stem + ".txt")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(txt)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
