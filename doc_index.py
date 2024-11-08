# indexer.py
# indexer.py
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
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
last_index_time =0

# Set up the text splitter
Settings.text_splitter = SentenceSplitter(chunk_size=400)

def doc_index():
    global global_query_engine, last_index_time

    try:
        logger.debug("Starting document indexing process.")
        documents = SimpleDirectoryReader(kb_dir, required_exts=['.md']).load_data()
        logger.debug(f"Number of documents loaded: {len(documents)}")

        if not documents:
            logger.info("No documents were processed for indexing.")
            return "No documents available to index."

        # Check for document modifications
        current_time = time.time()
        modified = False
        for doc in documents:
            file_time = os.path.getmtime(doc.metadata['file_path'])
            if file_time > last_index_time:
                modified = True
                break

        if not modified and global_query_engine:
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
            storage_context=storage_context,
            index_id="my_rag_index"  # Use a unique index ID 
        )

        logger.debug("Persisting index...")

        # Choose ONE of the following persistence options:

        # Option 1: Save to disk (recommended)
        index_file_path = os.path.join(kb_dir, "my_rag_index.json")
        index.save_to_disk(index_file_path)

        try:
            reloaded_storage_context = StorageContext.from_defaults(persist_dir=kb_dir)

            # Load using the correct index ID and file path (if applicable)
            reloaded_index = load_index_from_storage(
                reloaded_storage_context, 
                index_id="my_rag_index",  # Match the ID used during creation
                index_file=index_file_path  # Use this if you chose Option 1
            )
            logger.debug("Index successfully reloaded from storage.")
        except Exception as e:
            logger.exception(f"Failed to reload index from storage: {e}")
            return f"Index created but failed to reload: {str(e)}"
