# app.py

import gradio as gr
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from doc_loader import load_documents
from nemoguardrails import LLMRails, RailsConfig
import logging
import asyncio
import os
from llama_index.core.node_parser import SentenceSplitter


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set LLM and Embedding Model
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
Settings.text_splitter = SentenceSplitter(chunk_size=400, chunk_overlap=20)


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
            # Use SentenceSplitter to split the document into nodes
            parser = SentenceSplitter()
            nodes = parser.get_nodes_from_documents([content])  # content is wrapped in a list as expected by get_nodes_from_documents
            processed_documents.extend(nodes)  # Extend the list with nodes instead of creating new Node objects
            
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


# create stream_response

async def stream_response(query, history):
    if not global_query_engine:
        # Instead of returning, we yield the message since this is an async generator
        yield ("System", "Please load documents first.")
        return  # Stop iteration here

    # Yield history if it exists
    if history:
        yield history

    # Initialize guardrails for each query
    config = RailsConfig.from_path("./Config")
    rails = LLMRails(config)

    user_message = {"role": "user", "content": query}
    try:
        # Generate response using guardrails and the RAG system
        result = await rails.generate_async(messages=[user_message])
        
        # Process the result
        if isinstance(result, dict):
            if "content" in result:
                history.append((query, result["content"]))
                yield history  # Yielding the updated history
            else:
                history.append((query, str(result)))
                yield history  # Yielding the updated history
        elif isinstance(result, str):
            history.append((query, result))
            yield history  # Yielding the updated history
        elif hasattr(result, '__iter__'):
            for chunk in result:
                if isinstance(chunk, dict) and "content" in chunk:
                    history.append((query, chunk["content"]))
                    yield history  # Yielding the updated history
                else:
                    history.append((query, chunk))
                    yield history  # Yielding the updated history
        else:
            logger.error(f"Unexpected result type: {type(result)}")
            history.append((query, "Unexpected response format."))
            yield history  # Yielding the updated history

    except Exception as e:
        logger.error(f"An error occurred while generating response: {e}")
        history.append((query, "An error occurred while processing your request."))
        yield history  # Yielding the error message

# create Gradio UI and launch UI

import gradio as gr

def start_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("# RAG Chatbot for PDF Files")
        
        file_input = gr.File(label="Select files to upload", file_count="multiple")
        load_btn = gr.Button("Click to Load Documents")
        clear_docs_btn = gr.Button("Clear Documents")
        load_output = gr.Textbox(label="Load Status")
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Enter your question")
        clear_chat_btn = gr.Button("Clear Chat History")
        clear_all_btn = gr.Button("Clear All")

        # Function to load documents and create index
        load_btn.click(
            lambda x: (load_documents(*x), doc_index()),
            inputs=[file_input],
            outputs=[load_output, gr.State()]
        )

        # Function to reset documents
        def reset_documents():
            global global_query_engine
            global_query_engine = None
            return None, "Documents cleared"

        clear_docs_btn.click(
            reset_documents, 
            outputs=[file_input, load_output]
        )

        msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot])
        clear_chat_btn.click(lambda: [], outputs=[chatbot])
        clear_all_btn.click(lambda: None, None, chatbot, queue=False)

    demo.queue().launch(share=True, debug=True)

if __name__ == "__main__":
    start_gradio()
  
