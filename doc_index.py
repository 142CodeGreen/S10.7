# indexer.py
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
import logging
import os
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

# Configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

kb_dir = "./Config/kb"
global_query_engine = None

# Set up the text splitter
Settings.text_splitter = SentenceSplitter(chunk_size=400)

def doc_index():
    global global_query_engine

    try:
        # Use SimpleDirectoryReader to load documents
        documents = SimpleDirectoryReader(kb_dir, required_exts=['.md']).load_data()
        
        if not documents:
            logger.info("No documents were processed for indexing.")
            return "No documents available to index."

        # Check if there's an existing index
        if global_query_engine:
            logger.info("Index is up-to-date. No action taken.")
            return "Index is up-to-date."

        # If there's no query engine or if we want to ensure the index is up-to-date or rebuilt:
        vector_store = MilvusVectorStore(
            uri="milvus_demo.db",
            dim=1024,
            overwrite=False  # Don't overwrite existing data
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create or update the index
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )
        
        # Persist the index
        index.storage_context.persist(persist_dir=kb_dir)

        # Create the query engine
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
        global_query_engine = query_engine  # Update the global variable

        # Log that indexing was successful
        logger.info("Documents indexed successfully.")
        return "Documents indexed successfully."

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return f"Failed to index documents: {str(e)}"
