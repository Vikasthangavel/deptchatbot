# Dept Chatbot — AI Web Intelligence Agent

A Flask-based RAG (Retrieval-Augmented Generation) chatbot that lets you point it at **any public website**, instantly indexes the page content into a vector database, and answers your questions using only the knowledge found on that site.

---

## ✨ Features

- 🔗 **URL-based knowledge ingestion** — paste any URL and the app scrapes, chunks, and embeds the content automatically
- 🧠 **RAG pipeline** — powered by LangChain + ChromaDB + `all-MiniLM-L6-v2` embeddings
- 🤖 **GPT-4o-mini via OpenRouter** — fast, cost-effective LLM responses grounded in your site's content
- 🎛️ **Adjustable similarity threshold** — tune retrieval precision from 0 (all results) to 1 (exact matches only)
- 💬 **Real-time chat UI** — async AJAX chat with animated typing indicators; no page reloads
- 🔒 **Session isolation** — each browser session maintains its own vector index and chat history
- 📱 **Responsive design** — works cleanly on desktop and mobile

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask |
| LLM | GPT-4o-mini (via OpenRouter API) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (in-memory per session) |
| Orchestration | LangChain |
| Web Scraping | LangChain `WebBaseLoader` + BeautifulSoup4 |
| Frontend | Vanilla HTML/CSS/JS, Plus Jakarta Sans |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Vikasthangavel/deptchatbot.git
cd deptchatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
FLASK_SECRET_KEY=your_random_secret_key_here
```

> Get your free OpenRouter API key at [openrouter.ai](https://openrouter.ai).

### 5. Run the App

```bash
python app.py
```

Open your browser and visit **http://localhost:5000**

---

## 📖 How to Use

1. **Enter a URL** — paste any public website URL into the "Target Website URL" field
2. **Set a threshold** *(optional)* — adjust the similarity score threshold (0.0 = return all results, 0.5+ = stricter relevance filtering)
3. **Click "Scan Website"** — the app scrapes and indexes the page (first query may take a few seconds)
4. **Ask questions** — type your question and hit Send; answers are grounded exclusively in the scanned page

---

## 📁 Project Structure

```
dept chatbot/
├── app.py                  # Main Flask application & RAG logic
├── templates/
│   └── index.html          # Single-page chat UI
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── .gitignore
└── README.md
```

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | Your OpenRouter API key for LLM access |
| `FLASK_SECRET_KEY` | Secret key for Flask session signing |

---

## 📦 Dependencies

```
langchain>=0.1.0
langchain-openai>=0.1.0
langchain-community>=0.0.20
chromadb>=0.4.22
beautifulsoup4
requests
sentence-transformers
Flask>=3.0.0
python-dotenv>=1.0.0
```

---

## ⚙️ How It Works

```
URL Input
   │
   ▼
WebBaseLoader (scrape page)
   │
   ▼
RecursiveCharacterTextSplitter (chunk_size=800, overlap=100)
   │
   ▼
HuggingFaceEmbeddings (all-MiniLM-L6-v2)
   │
   ▼
ChromaDB (in-memory vector store)
   │
   ▼
Similarity Retrieval (cosine distance, configurable threshold)
   │
   ▼
ChatPromptTemplate + GPT-4o-mini (via OpenRouter)
   │
   ▼
Answer → Chat UI (AJAX)
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

> Built with ❤️ using Flask, LangChain, and ChromaDB.
