# app.py

import gradio as gr
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from doc_loader import load_documents
from indexer import doc_index
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

def process_result(result):
    if isinstance(result, dict):
        return result.get('content', str(result))
    elif isinstance(result, str):
        return result
    elif hasattr(result, '__iter__'):
        return next((chunk['content'] for chunk in result if isinstance(chunk, dict) and 'content' in chunk), "")
    else:
        logger.error(f"Unexpected result type: {type(result)}")
        return "Unexpected response format."


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
  
