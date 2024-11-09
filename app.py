# app.py

import gradio as gr
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from doc_loader import load_documents
from doc_index import doc_index
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

rails = None  # Global variable for rails, if needed

def initialize_guardrails(query_engine):
    if query_engine:
        config = RailsConfig.from_path("./Config")
        global rails  # Use global to update the rails variable
        rails = LLMRails(config)
        init(rails)
        return "Guardrails initialized successfully.", rails
    else:
        error_message = "Guardrails not initialized: Query engine is None."
        logger.error(error_message)
        return error_message, None

async def stream_response(query, history):
    global rails  # Use global to access the rails variable
    if not rails:
        logger.error("Guardrails not initialized.")
        yield [("System", "Guardrails not initialized. Please load documents first.")]
        return

    try:
        user_message = {"role": "user", "content": query}
        result = await rails.generate_async(messages=[user_message])

        if isinstance(result, dict):
            if "content" in result:
                history.append((query, result["content"]))
            else:
                history.append((query, str(result)))
        else:
            if isinstance(result, str):
                history.append((query, result))
            elif hasattr(result, '__iter__'):
                for chunk in result:
                    if isinstance(chunk, dict) and "content" in chunk:
                        history.append((query, chunk["content"]))
                        yield history
                    else:
                        history.append((query, chunk))
                        yield history
            else:
                logger.error(f"Unexpected result type: {type(result)}")
                history.append((query, "Unexpected response format."))

        yield history

    except Exception as e:
        logger.error(f"Error in stream_response: {str(e)}")
        history.append(("An error occurred while processing your query.", None))
        yield history

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

        load_btn.click(
            lambda x: [load_documents(*x), *doc_index()],
            inputs=[file_input],
            outputs=[load_output, gr.Textbox(label="Index Status")]
        ).then(
            initialize_guardrails,
            inputs=[gr.Textbox()],  # Assumes the query_engine is passed here
            outputs=[gr.Textbox(label="Guardrail Status")]
        )

        # Function to clear documents is no longer needed; use the small "x" sign at the file input

        clear_docs_btn.click(
            lambda: ([], None, "Documents cleared"),
            inputs=[], 
            outputs=[file_input, load_output]
        )

        msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot])
        clear_chat_btn.click(lambda: [], outputs=[chatbot])
        clear_all_btn.click(lambda: ([], None, "Documents and chat cleared"), inputs=[], outputs=[chatbot, file_input, load_output])

    demo.queue().launch(share=True, debug=True)

if __name__ == "__main__":
    start_gradio()
