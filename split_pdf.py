#!/usr/bin/env python3
import sys
try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    from PyPDF2 import PdfFileReader as PdfReader, PdfFileWriter as PdfWriter
import os

def split_pdf(input_pdf, chunk_size=10):
    """
    Split a PDF into chunks of specified size
    
    Args:
        input_pdf: Path to input PDF file
        chunk_size: Number of pages per chunk (default: 10)
    """
    reader = PdfReader(input_pdf)
    try:
        total_pages = len(reader.pages)
    except AttributeError:
        total_pages = reader.numPages
    
    base_name = os.path.splitext(os.path.basename(input_pdf))[0]
    output_dir = os.path.dirname(input_pdf)
    if not output_dir:
        output_dir = "."
    
    print(f"Splitting {input_pdf}")
    print(f"Total pages: {total_pages}")
    print(f"Chunk size: {chunk_size} pages")
    print()
    
    chunk_num = 1
    start_page = 0
    
    while start_page < total_pages:
        end_page = min(start_page + chunk_size, total_pages)
        
        writer = PdfWriter()
        
        for page_num in range(start_page, end_page):
            try:
                writer.add_page(reader.pages[page_num])
            except AttributeError:
                writer.addPage(reader.getPage(page_num))
        
        output_filename = f"{base_name}_pages_{start_page+1}-{end_page}.pdf"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        print(f"Created: {output_filename} (pages {start_page+1}-{end_page})")
        
        start_page = end_page
        chunk_num += 1
    
    print()
    print(f"Done! Created {chunk_num-1} chunks")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_pdf.py <pdf_file> [chunk_size]")
        print("Example: python split_pdf.py lecture.pdf 10")
        sys.exit(1)
    
    input_pdf = sys.argv[1]
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    if not os.path.exists(input_pdf):
        print(f"Error: File '{input_pdf}' not found")
        sys.exit(1)
    
    split_pdf(input_pdf, chunk_size)
