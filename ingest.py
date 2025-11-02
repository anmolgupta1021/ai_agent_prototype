#!/usr/bin/env python3
"""ingest.py - simple PDF to text using PyMuPDF (fitz)
Usage:
    python ingest.py --pdf path/to/file.pdf --output out.txt
"""

import argparse
import fitz  # pymupdf

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pdf', type=str, required=True)
    p.add_argument('--output', type=str, default='out.txt')
    return p.parse_args()

def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return '\n\n'.join(pages)

def main():
    args = parse_args()
    text = pdf_to_text(args.pdf)
    with open(args.output,'w', encoding='utf-8') as f:
        f.write(text)
    print(f'Wrote extracted text to {args.output}')

if __name__ == '__main__':
    main()
