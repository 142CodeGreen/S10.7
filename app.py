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
#rails = None


# create stream_response
async def stream_response(query, history):
    if not global_query_engine:
        yield [("System", "Please load documents first.")]
        yield history

    # Initialize guardrails for each query
    config = RailsConfig.from_path("./Config")
    rails = LLMRails(config)
    init(rails)

    try:
        user_message = {"role": "user", "content": query}
        # Generate response using guardrails and the RAG system
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
            lambda x: [
                load_documents(*x),
                (
            "Indexing completed." 
            if doc_index() is None  # Assuming doc_index() returns None on success
            else "Indexing failed." 
                )
            ],
            inputs=[file_input],
            outputs=[load_output, gr.State()]
        )
        
        #load_btn.click(
        #    lambda x: [load_documents(*x), doc_index()],
        #    inputs=[file_input],
        #    outputs=[load_output, gr.State()]
        #)

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
        clear_all_btn.click(lambda: ([], None, "Documents and chat cleared"), inputs=[], outputs=[chatbot, file_input, load_output])

    demo.queue().launch(share=True, debug=True)

if __name__ == "__main__":
    start_gradio()
  
