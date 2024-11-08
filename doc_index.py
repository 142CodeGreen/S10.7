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
#global_query_engine = None

# Set up the text splitter
Settings.text_splitter = SentenceSplitter(chunk_size=400)

def doc_index():
    #global global_query_engine
    try:
        logger.debug("Starting document indexing process.")
        documents = SimpleDirectoryReader(kb_dir, required_exts=['.md']).load_data()
        logger.debug(f"Number of documents loaded: {len(documents)}")

        if not documents:
            logger.info("No documents were processed for indexing.")
            return "No documents available to index."

        vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Create the query engine DIRECTLY from the index
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True) 

        logger.info("Documents indexed successfully.")
        return query_engine, "Documents indexed successfully

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return f"Failed to index documents: {str(e)}"
