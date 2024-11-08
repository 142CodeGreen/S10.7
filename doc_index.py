# indexer.py
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index import load_index_from_storage
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

        if global_query_engine:
            logger.info("Index is up-to-date. No action taken.")
            return "Index is up-to-date."

        vector_store = MilvusVectorStore(
            uri="milvus_demo.db",
            dim=1024,
            overwrite=True
        )
        logger.debug(f"Milvus connection: {vector_store.uri}")

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        logger.debug("Creating VectorStoreIndex...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )        
        logger.debug("Persisting index...")
        index.storage_context.persist(persist_dir=kb_dir)

        try:
            reloaded_storage_context = StorageContext.from_defaults(persist_dir=kb_dir)
            reloaded_index = load_index_from_storage(reloaded_storage_context)
            logger.debug("Index successfully reloaded from storage.")
        except Exception as e:
            logger.error(f"Failed to reload index from storage: {e}")
            return f"Index created but failed to reload: {str(e)}"

        logger.debug("Creating query engine...")
        global_query_engine = reloaded_index.as_query_engine(similarity_top_k=20, streaming=True)
        logger.info("Documents indexed successfully.")
        return "Documents indexed successfully."

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return f"Failed to index documents: {str(e)}"
