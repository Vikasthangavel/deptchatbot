import os
import uuid

from dotenv import load_dotenv
from flask import Flask, render_template, request, session, jsonify

# Load .env file before anything else
load_dotenv()

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

chat_sessions = {}

# Shared LLM (initialised once at startup)
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0,
)

prompt = ChatPromptTemplate.from_template("""
You are an AI assistant for a website provided by a user.

Answer user questions based only on the website information below.

Context:
{context}

Question:
{question}

Give clear, helpful answers based on the site content.
""")


def build_vectorstore_for_url(url: str):
    """Load a URL, build embeddings and a Chroma vectorstore."""
    loader = WebBaseLoader([url])
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore


def build_rag_chain_for_vectorstore(vectorstore, threshold: float):
    """Build a RAG chain for a given vectorstore and similarity threshold."""
    if threshold <= 0.0:
        retriever = vectorstore.as_retriever()
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": threshold}
        )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def get_session_state():
    session_id = session.get("chat_session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["chat_session_id"] = session_id

    return chat_sessions.setdefault(
        session_id,
        {"url": None, "vectorstore": None, "rag_chain": None, "threshold": 0.0, "messages": []},
    )


@app.route("/", methods=["GET", "POST"])
def index():
    state = get_session_state()
    error = None

    if request.method == "POST":
        action = request.form.get("action", "load")
        url = request.form.get("url", "").strip()

        if action == "load":
            if not url:
                error = "Please paste a website URL."
            else:
                try:
                    threshold = float(request.form.get("threshold", "0.0"))
                except ValueError:
                    threshold = 0.0
                # Reset session for new URL
                state["url"] = url
                state["vectorstore"] = None
                state["rag_chain"] = None
                state["threshold"] = threshold
                state["messages"] = []

    return render_template(
        "index.html",
        error=error,
        current_url=state.get("url"),
        current_threshold=state.get("threshold", 0.0),
        chat_history=state.get("messages", []),
    )


@app.route("/chat", methods=["POST"])
def chat():
    """
    JSON endpoint for the chat AJAX call.
    Returns: {"answer": "...", "error": "..."}
    This keeps the slow RAG work off the page-render route and prevents
    ERR_CONNECTION_RESET caused by the debug reloader interrupting long requests.
    """
    state = get_session_state()

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    url = (data.get("url") or "").strip()
    try:
        threshold = float(data.get("threshold", 0.0))
    except ValueError:
        threshold = 0.0

    # Allow the client to pass the URL/threshold in case the session was lost
    if not state.get("url") and url:
        state["url"] = url
        state["vectorstore"] = None
        state["rag_chain"] = None
        state["threshold"] = threshold
        state["messages"] = []

    if not state.get("url"):
        return jsonify({"error": "Please load a website URL first."})

    if not question:
        return jsonify({"error": "Please enter a question."})

    try:
        if state.get("vectorstore") is None:
            state["vectorstore"] = build_vectorstore_for_url(state["url"])

        # Rebuild the chain if it is None or if the threshold has changed
        if state.get("rag_chain") is None or state.get("threshold") != threshold:
            state["threshold"] = threshold
            state["rag_chain"] = build_rag_chain_for_vectorstore(state["vectorstore"], threshold)

        answer = state["rag_chain"].invoke(question)
        state.setdefault("messages", []).append({"role": "user", "text": question})
        state.setdefault("messages", []).append({"role": "assistant", "text": answer})
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    # threaded=True lets Flask handle each request in its own thread,
    # which prevents the debug reloader from resetting long-running connections.
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)