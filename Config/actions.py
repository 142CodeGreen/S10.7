#actions.py

from app import global_query_engine

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.actions.actions import action
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core import Settings
import asyncio

# Settings for LLM and embedding model
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

def template(question, context, history):
    """Constructs a prompt template for the RAG system, including conversation history."""
    history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
    return f"""Answer user questions based on loaded documents and past conversation.

    Past conversation:
    {history_str}

    Current Context:
    {context}

    1. Use the information above to answer the question.
    2. You do not make up a story.
    3. Keep your answer as concise as possible.
    4. Should not answer any out-of-context USER QUESTION.

    USER QUESTION: ```{question}```
    Answer in markdown:"""

@action(is_system_action=True)
async def rag(context: dict, embed_model: NVIDIAEmbedding, query_engine):
    """
    Implements RAG functionality by querying the global_query_engine with the user's question,
    considering the conversation history.

    Args:
    context (dict): A dictionary containing 'last_user_message' and 'history'.

    Returns:
    ActionResult: An action result containing the generated answer and updated context.
    """
    print("rag() function called!")
    print(f"Received context: {context}")
    context_updates = {}
    question = context.get('last_user_message', '')
    history = context.get('history', [])

    if global_query_engine is None:
        return ActionResult(
            return_value="No documents have been indexed. Please load documents first.",
            context_updates={}
        )

    try:
        # Retrieve relevant contexts using the global_query_engine
        response = await global_query_engine.aquery(question)
        
        # Create context from retrieved documents
        doc_context = "\n".join([node.text for node in response.source_nodes])
        
        # Use the template to form the prompt including history
        prompt = template(question, doc_context, history)

        # Generate the response using the LLM
        answer = await Settings.llm.complete(prompt)

        # Update context with new information
        context_updates = {
            "relevant_chunks": doc_context,
            "history": history + [(question, answer.text)]  # Update history
        }

        return ActionResult(
            return_value=answer.text,
            context_updates=context_updates
        )
    except Exception as e:
        print(f"Error in rag(): {e}")
        return ActionResult(
            return_value="An error occurred while processing your query.",
            context_updates={}
        )

def init(app: LLMRails):
    app.register_action(
        rag, 
        name="rag", 
        embed_model=Settings.embed_model,
        query_engine=global_query_engine  # Pass the query engine
    )
