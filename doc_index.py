# indexer.py

import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
import logging
from doc_loader import load_documents

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

kb_dir = "./Config/kb"
global_query_engine = None

def doc_index():
    global global_query_engine
    markdown_files = [os.path.join(kb_dir, f) for f in os.listdir(kb_dir) if f.endswith('.md')]
    
    if not markdown_files:
        logger.info("No Markdown files found in knowledge base directory. Please load documents first.")
        return "No documents available to index."

    if global_query_engine is not None:
        logger.warning("Attempting to index documents while there's already an index. This should not happen unless it's an intentional reload.")
        return "An index already exists. Reload documents if you want to update the index."

    # Load documents using SimpleDirectoryReader
    documents = SimpleDirectoryReader(kb_dir, required_exts=['.md']).load_data()
    if documents:
        try:
            vector_store = MilvusVectorStore(
                uri="milvus_demo.db",
                dim=1024,
                overwrite=True
            )

            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context
            )
            # Persist the index
            index.storage_context.persist(persist_dir=kb_dir)
            query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
            global_query_engine = query_engine  # Set the global variable

            # Sample query to confirm the query engine works
            response = query_engine.query("What is this document about?")
            logger.info("Sample query response: " + str(response))

            logger.info("Documents indexed successfully.")
            return "Documents indexed successfully."
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            return "Failed to index documents."
    else:
        return "No documents were processed for indexing."
