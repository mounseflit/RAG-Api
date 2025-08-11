import os
import tempfile
import shutil
from typing import List, Dict, Any, Optional

# Text processing
import pypdf
from docx import Document
import pandas as pd

# OCR and image processing
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# PDF to image conversion
try:
    from pdf2image import convert_from_path
    from PIL import ImageEnhance
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False



def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    if not text or len(text.strip()) == 0:
        return []
    
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        chunks.append(chunk_text)
        
        if end >= len(words):
            break
            
        start = end - overlap
    
    return chunks



def process_text_file(file_path: str) -> str:
    """Process plain text files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(file_path, 'r', encoding='latin-1') as file:
            return file.read()


# def process_pdf_file(file_path: str) -> str:
#     """Process PDF files with fallback to OCR for scanned documents."""
#     text = ""
    
#     try:
#         # First, try to extract text directly
#         with open(file_path, 'rb') as file:
#             pdf_reader = pypdf.PdfReader(file)
#             for page in pdf_reader.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text + "\n"
        
#         # If we got very little text, it might be a scanned PDF
#         if len(text.strip()) < 100 and PDF2IMAGE_AVAILABLE and OCR_AVAILABLE:
#             print("Low text content detected, attempting OCR...")
#             try:
#                 # Convert PDF to images
#                 images = convert_from_path(file_path)
#                 ocr_text = ""
                
#                 for i, image in enumerate(images):
#                     # Save image temporarily
#                     with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
#                         image.save(tmp_img.name, 'PNG')
                        
#                         # Perform OCR
#                         page_text = pytesseract.image_to_string(Image.open(tmp_img.name))
#                         ocr_text += f"Page {i+1}:\n{page_text}\n\n"
                        
#                         # Clean up
#                         os.unlink(tmp_img.name)
                
#                 if len(ocr_text.strip()) > len(text.strip()):
#                     text = ocr_text
#                     print("OCR extraction successful")
                    
#             except Exception as e:
#                 print(f"OCR failed: {e}")
                
#     except Exception as e:
#         print(f"PDF processing error: {e}")
#         return ""
    
#     return text


def process_pdf_file(file_path: str) -> str:
    """Process PDF files with fallback to OCR for scanned documents."""
    text = ""
    
    try:
        # First, try to extract text directly
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # If we got little text or PDF seems to be scanned, try OCR
        if (len(text.strip()) < 200 or text.count(' ') < 50) and PDF2IMAGE_AVAILABLE and OCR_AVAILABLE:
            print(f"Low text content detected in {os.path.basename(file_path)}, attempting OCR...")
            try:
                # Create temp directory for images
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Convert PDF to images with higher DPI for better OCR results
                    images = convert_from_path(
                        file_path, 
                        dpi=300,
                        output_folder=temp_dir,
                        fmt='png',
                        thread_count=4
                    )
                    ocr_text = ""
                    
                    # Configure OCR for better results
                    custom_config = r'--oem 3 --psm 6 -l eng+osd'
                    
                    for i, image in enumerate(images):
                        image_path = os.path.join(temp_dir, f'page_{i+1}.png')
                        image.save(image_path, 'PNG')
                        
                        # Try to improve image quality before OCR
                        img = Image.open(image_path)
                        # Apply contrast enhancement for clearer text
                        enhancer = ImageEnhance.Contrast(img)
                        img = enhancer.enhance(1.5)  # Increase contrast
                        
                        # Perform OCR with custom configuration
                        page_text = pytesseract.image_to_string(img, config=custom_config)
                        ocr_text += f"Page {i+1}:\n{page_text}\n\n"
                
                if len(ocr_text.strip()) > max(50, len(text.strip())):
                    text = ocr_text
                    print(f"OCR extraction successful: extracted {len(text)} characters")
                    
            except Exception as e:
                print(f"OCR failed: {str(e)}")
                # If OCR failed but we have some text from direct extraction, use that
                if len(text.strip()) < 50:
                    print("Falling back to minimal text from direct extraction")
                
    except Exception as e:
        print(f"PDF processing error: {str(e)}")
        return ""
    
    return text



def process_docx_file(file_path: str) -> str:
    """Process DOCX files."""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"DOCX processing error: {e}")
        return ""



def process_excel_file(file_path: str) -> str:
    """Process Excel files."""
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        all_text = ""
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Convert DataFrame to text
            sheet_text = f"Sheet: {sheet_name}\n"
            sheet_text += df.to_string(index=False)
            sheet_text += "\n\n"
            
            all_text += sheet_text
        
        return all_text
    except Exception as e:
        print(f"Excel processing error: {e}")
        return ""



def process_image_file(file_path: str) -> str:
    """Process image files using OCR."""
    if not OCR_AVAILABLE:
        return "OCR not available - pytesseract not installed"
    
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Image OCR error: {e}")
        return ""




def process_file(file_path: str, file_type: str) -> List[Dict[str, Any]]:
    """
    Process a file and return structured data for RAG.
    
    Args:
        file_path: Path to the file
        file_type: Type of file ('txt', 'pdf', 'docx', 'xlsx', 'image')
    
    Returns:
        List of dictionaries containing processed content and metadata
    """
    
    # Extract text based on file type
    if file_type == 'txt':
        text = process_text_file(file_path)
    elif file_type == 'pdf':
        text = process_pdf_file(file_path)
    elif file_type == 'docx':
        text = process_docx_file(file_path)
    elif file_type == 'xlsx':
        text = process_excel_file(file_path)
    elif file_type == 'image':
        text = process_image_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    if not text or len(text.strip()) == 0:
        return []
    
    # Chunk the text
    chunks = chunk_text(text)
    
    # Create structured data
    processed_data = []
    file_name = os.path.basename(file_path)
    
    for i, chunk in enumerate(chunks):
        if chunk.strip():  # Only include non-empty chunks
            item = {
                "content": chunk.strip(),
                "metadata": {
                    "file_name": file_name,
                    "file_type": file_type,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk),
                    "word_count": len(chunk.split())
                }
            }
            processed_data.append(item)
    
    return processed_data