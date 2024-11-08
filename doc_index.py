# indexer.py

import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
import logging
from doc_loader import load_documents

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

kb_dir = "./Config/kb"
global_query_engine = None

def doc_index():
    global global_query_engine

    # First, ensure documents are loaded
    load_result = load_documents(*os.listdir(kb_dir))  # Assuming all files in kb_dir are PDFs
    if load_result != "Documents successfully converted to Markdown and saved.":
        logger.info("Document loading failed or no documents were processed.")
        return load_result

    markdown_files = [os.path.join(kb_dir, f) for f in os.listdir(kb_dir) if f.endswith('.md')]
    
    if not markdown_files:
        logger.info("No Markdown files found in knowledge base directory. Please load documents first.")
        return "No documents available to index."

    try:
        # Load documents using SimpleDirectoryReader
        documents = SimpleDirectoryReader(kb_dir, required_exts=['.md']).load_data()
        if not documents:
            return "No documents were processed for indexing."

        # Check if there's an existing index
        if global_query_engine:
            current_docs_count = len(documents)
            if current_docs_count == len(global_query_engine._index.index_struct.docstore.docs):
                logger.info("No new documents to index. Using existing index.")
                return "Index is up-to-date. No action taken."

        # If there's no query engine or if documents have changed, create a new index or update
        vector_store = MilvusVectorStore(
            uri="milvus_demo.db",
            dim=1024,
            overwrite=False  # Don't overwrite, but this doesn't mean we can't update or add
        )

        storage_context = StorageContext.from_defaults(persist_dir=kb_dir, vector_store=vector_store)

        # If we have an existing index, try to load it, otherwise create a new one
        if global_query_engine:
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        else:
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context
            )
            index.storage_context.persist(persist_dir=kb_dir)

        # Create the query engine
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
        global_query_engine = query_engine  # Set or update the global variable

        # Sample query to confirm the query engine works
        response = query_engine.query("What is this document about?")
        logger.info("Sample query response: " + str(response))

        logger.info("Documents indexed successfully.")
        return "Documents indexed successfully."

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return "Failed to index documents."
