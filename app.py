import os
import uuid

from dotenv import load_dotenv
from flask import Flask, render_template, request, session

# OpenRouter API key is loaded from .env or the environment.
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

# Shared components (LLM + prompt)
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0
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


def build_rag_chain_for_url(url: str):
    """Load a URL, build embeddings and a RAG chain for it."""
    loader = WebBaseLoader([url])
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create an in-memory Chroma vectorstore for this request
    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

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
        {"url": None, "rag_chain": None, "messages": []},
    )


@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    error = None
    current_url = None
    chat_history = []
    state = get_session_state()

    if request.method == "POST":
        action = request.form.get("action", "ask")
        url = request.form.get("url", "").strip()
        question = request.form.get("question", "").strip()

        if action == "load":
            if not url:
                error = "Please paste a website URL."
            else:
                state["url"] = url
                state["rag_chain"] = None
                state["messages"] = []
                current_url = state["url"]

        elif action == "ask":
            if not state["url"]:
                error = "Please paste a website URL first."
            elif not question:
                error = "Please enter a question."
            else:
                try:
                    if state["rag_chain"] is None:
                        state["rag_chain"] = build_rag_chain_for_url(state["url"])

                    answer = state["rag_chain"].invoke(question)
                    state["messages"].append({"role": "user", "text": question})
                    state["messages"].append({"role": "assistant", "text": answer})
                except Exception as e:
                    error = str(e)

        current_url = state["url"]
        chat_history = state["messages"]

    else:
        current_url = state["url"]
        chat_history = state["messages"]

    return render_template(
        "index.html",
        answer=answer,
        error=error,
        current_url=current_url,
        chat_history=chat_history,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)