import os
import fitz  # PyMuPDF
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pdf_extractor')

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        doc = fitz.open(pdf_path)
        text = []
        
        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text.append(page.get_text())
            
        doc.close()
        return "\n\n".join(text)
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return f"ERROR EXTRACTING TEXT FROM {pdf_path}: {str(e)}"

def main():
    # Directory containing the PDF files
    pdf_directory = "assets"
    # Output file for combined text
    output_file = "honeybadger.txt"
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_directory}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_directory}")
    
    # Extract text from each PDF and combine
    all_text = []
    
    for pdf_file in tqdm(pdf_files, desc="Extracting text from PDFs"):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        logger.info(f"Processing {pdf_file}")
        
        # Add file name as header
        file_header = f"\n\n{'='*80}\n{pdf_file}\n{'='*80}\n\n"
        all_text.append(file_header)
        
        # Extract and add text
        pdf_text = extract_text_from_pdf(pdf_path)
        all_text.append(pdf_text)
    
    # Write combined text to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_text))
    
    logger.info(f"Text extraction complete. Combined text saved to {output_file}")
    print(f"Text extraction complete. Combined text saved to {output_file}")

if __name__ == "__main__":
    main() 