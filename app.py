# app.py

import gradio as gr
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from doc_loader import load_documents
from doc_index import doc_index, global_query_engine
from Config.actions import init
from nemoguardrails import LLMRails, RailsConfig
import logging
import asyncio
import os


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set LLM and Embedding Model
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
Settings.text_splitter = SentenceSplitter(chunk_size=400, chunk_overlap=20)


kb_dir = "./Config/kb"
global_query_engine = None
rails = None

def initialize_guardrails():
    global rails
    if global_query_engine:
        try:
            config = RailsConfig.from_path("./Config")
            rails = LLMRails(config)
            init(rails)  # This calls the init function from actions.py
            logger.info("Guardrails initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize guardrails: {e}")
            return "Failed to initialize guardrails."
    else:
        logger.warning("Attempt to initialize guardrails without query engine.")
        return "Query engine not available. Please index documents first."

def load_and_initialize(*file_paths):
    # Step 1: Upload documents
    load_result = load_documents(*file_paths)
    
    # If the documents are loaded successfully, proceed with indexing
    if load_result == "Documents successfully converted to Markdown and saved.":
        # Step 2: Indexing
        index_result = doc_index()
        
        # If indexing was successful, initialize guardrails
        if index_result == "Documents indexed successfully.":
            # Step 3: Initialize guardrails
            guardrail_result = initialize_guardrails()
            return guardrail_result
        else:
            return index_result
    else:
        return load_result


# create stream_response

async def stream_response(query, history):
    if not global_query_engine:
        yield ("System", "Please load documents first.")
        return  # Stop iteration here

    if not rails:
        yield ("System", "Guardrails have not been initialized. Please index documents first.")
        return

    # Yield history if it exists
    if history:
        yield history

    try:
        user_message = {"role": "user", "content": query}
        # Generate response using guardrails and the RAG system
        result = await rails.generate_async(messages=[user_message])
        
        # Process the result
        response = process_result(result)
        history.append((query, response))
        yield history

    except Exception as e:
        logger.error(f"An error occurred while generating response: {e}")
        history.append((query, "An error occurred while processing your request."))
        yield history

def load_and_initialize(*file_paths):
    # Step 1: Upload documents
    load_result = load_documents(*file_paths)
    
    # If the documents are loaded successfully, proceed with indexing
    if load_result == "Documents successfully converted to Markdown and saved.":
        # Step 2: Indexing
        index_result = doc_index()
        
        # If indexing was successful, initialize guardrails
        if index_result == "Documents indexed successfully.":
            # Step 3: Initialize guardrails
            guardrail_result = initialize_guardrails()
            return guardrail_result
        else:
            return index_result
    else:
        return load_result


# create Gradio UI and launch UI

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
            outputs=[load_output]
        )  #gr.State()]

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
  
