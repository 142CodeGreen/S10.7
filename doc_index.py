# indexer.py

import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
import logging
from doc_loader import load_documents
from llama_index.core.node_parser import SentenceSplitter
Settings.text_splitter = SentenceSplitter(chunk_size=400)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

kb_dir = "./Config/kb"
global_query_engine = None

def doc_index():
    global global_query_engine
    markdown_files = [os.path.join(kb_dir, f) for f in os.listdir(kb_dir) if f.endswith('.md')]

    if not markdown_files:
        print("No Markdown files found in knowledge base directory. Please load documents first.")
        return "No documents available to index."

    if global_query_engine is not None:
        logger.warning("Attempting to index documents while there's already an index. This should not happen unless it's an intentional reload.")
        return "An index already exists. Reload documents if you want to update the index."

    processed_documents = []
    for md_path in markdown_files:
        with open(md_path, 'r') as file:
            content = file.read()
            nodes = Settings.text_splitter.split_text(content)
            for node in nodes:
                processed_documents.append(Node(text=node, metadata={"source": md_path}))

    if processed_documents:
        # Create Milvus Vector Store with processed documents
        vector_store = MilvusVectorStore(
            uri="milvus_demo.db",
            dim=1024,
            overwrite=True
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            processed_documents,
            storage_context=storage_context
        )
        # Persist the index
        index.storage_context.persist(persist_dir=kb_dir)
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
        global_query_engine = query_engine  # Set the global variable

        # Sample query to confirm the query engine works
        response = query_engine.query("What is this document about?")
        print(response)

        return "Documents indexed successfully."
    else:
        return "No documents were processed for indexing."
