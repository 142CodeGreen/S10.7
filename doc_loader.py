#doc_loader.py, only convert and upload documents to kb
from functools import lru_cache
import os
from llama_index.readers.file import PDFReader
import logging

logger = logging.getLogger(__name__)

@lru_cache(maxsize=8)
def load_documents(*f_paths):
    """
    Converts PDF documents to Markdown format, saves them in a specified directory,
    and prints a message upon completion.

    This function does not index or process documents for use with search or retrieval systems.

    Args:
    *f_paths (str): Variable length argument list of file paths.

    Returns:
    list: A list of tuples containing the original document and its Markdown file path.
          If no documents are processed, an empty list is returned.
    """
    kb_dir = "./Config/kb"
    
    # Clear the cache before loading new documents
    load_documents.cache_clear()

    # Ensure the knowledge base directory exists
    try:
        os.makedirs(kb_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating directory {kb_dir}: {e}")
        return "Failed to create knowledge base directory."

    documents = []

    for file_path in f_paths:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            try:
                if file_path.lower().endswith(".pdf"):
                    reader = PDFReader()
                    docs = reader.load_data(file_path)
                    for i, doc in enumerate(docs):
                        markdown_filename = os.path.splitext(os.path.basename(file_path))[0] + f"_{i+1}.md"
                        markdown_filepath = os.path.join(kb_dir, markdown_filename)
                        with open(markdown_filepath, "w") as f:
                            f.write(doc.text)
                        documents.append((doc, markdown_filepath))
                else:
                    logger.info(f"Unsupported file format: {file_path}")
            except Exception as e:
                logger.error(f"Error converting document from {file_path}: {str(e)}")

    if not documents:
        logger.info("No documents were converted.")
        return "No documents were processed for conversion."
    else:
        logger.info(f"Documents successfully converted to Markdown and saved.")
        return f"{num_docs} documents successfully converted to Markdown and saved."
