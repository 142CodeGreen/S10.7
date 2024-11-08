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
        logger.debug("Starting document indexing process.")
        documents = SimpleDirectoryReader(kb_dir, required_exts=['.md']).load_data()
        logger.debug(f"Number of documents loaded: {len(documents)}")

        if not documents:
            logger.info("No documents were processed for indexing.")
            return "No documents available to index."

        # Create a Milvus vector store and storage context (using GPU)
        #vector_store = MilvusVectorStore(
        #    host="127.0.0.1",
        #    port=19530,
        #    dim=1024,
        #    collection_name="your_collection_name",  # Replace with your desired collection name
        #    gpu_id=0  # Specify the GPU ID to use
        #)

        # If you want to use CPU, uncomment the following line and comment out the GPU configuration above
        vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create the index from the documents
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Create the query engine DIRECTLY from the index
        global_query_engine = index.as_query_engine(similarity_top_k=20, streaming=True) 

        logger.info("Documents indexed successfully.")
        return "Documents indexed successfully."

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return f"Failed to index documents: {str(e)}"
