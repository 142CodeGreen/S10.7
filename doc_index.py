# indexer.py

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
import logging

# Configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

kb_dir = "./Config/kb"

# Set up the text splitter and embedding model
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.text_splitter = SentenceSplitter(chunk_size=400)
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

def doc_index():
    try:
        logger.debug("Starting document indexing process.")
        documents = SimpleDirectoryReader(kb_dir, required_exts=['.md']).load_data()
        logger.debug(f"Number of documents loaded: {len(documents)}")

        if not documents:
            logger.info("No documents were processed for indexing.")
            return None, "No documents available to index."

        vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Create the query engine with enhanced parameters
        query_engine = index.as_query_engine(
            similarity_top_k=20, 
            streaming=True,
            # Additional parameters can be added here as needed
        )

        logger.info("Documents indexed successfully.")
        return query_engine, "Documents indexed successfully"

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return None, f"Failed to index documents: {str(e)}"
