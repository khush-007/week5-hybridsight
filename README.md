# 🔭 HybridSight

## AI That Reads, Searches, and Sees

HybridSight is a hybrid AI agent that combines **Retrieval-Augmented Generation (RAG), web search, Wikipedia, and computer vision** to answer different types of user questions.

The system uses an LLM-based planner to determine which information source or tool is relevant to a user's question and then generates a final response from the retrieved information.

---

## 🚀 Live Demo

👉 **[Try HybridSight Live](https://khush-0007-hybridsight.hf.space)**

## 🚀 Features

- 📄 **PDF / RAG**
  - Upload PDF documents
  - Extract and split document content
  - Generate embeddings
  - Store document embeddings in ChromaDB
  - Retrieve relevant document chunks

- 🖼️ **Vision**
  - Upload images
  - Analyze images using a vision-capable Groq model
  - Answer questions about image content

- 🌐 **Web Search**
  - Search the web using DuckDuckGo
  - Useful for current and recent information

- 📚 **Wikipedia**
  - Search Wikipedia for general knowledge questions

- 🧠 **LLM-Based Tool Planning**
  - The LLM analyzes the user's question
  - Determines which tools are relevant
  - Executes the required tools
  - Combines retrieved information into a final response

- 🔗 **Hybrid AI**
  - Supports multiple information sources
  - Designed to combine results from different tools

- 💬 **Gradio Interface**
  - Interactive web interface
  - PDF upload
  - Image upload
  - Chat interface

---

## 🏗️ Architecture

```text
                         User
                          │
                          ▼
                  ┌───────────────┐
                  │  User Query   │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │   Planner LLM │
                  │               │
                  │ Decides which │
                  │ tools to use  │
                  └───────┬───────┘
                          │
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────────┐
        │ PDF/RAG │  │ Vision  │  │ Web / Wiki  │
        └────┬────┘  └────┬────┘  └──────┬──────┘
             │            │              │
             ▼            ▼              ▼
        ┌─────────┐  ┌─────────┐  ┌─────────────┐
        │ChromaDB │  │Groq     │  │Search APIs  │
        │         │  │Vision   │  │             │
        └────┬────┘  └────┬────┘  └──────┬──────┘
             │            │              │
             └────────────┼──────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Tool Results  │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Synthesis LLM │
                  │               │
                  │ Combines tool │
                  │ results       │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Final Answer  │
                  └───────────────┘
🛠️ Tech Stack
Technology	Purpose
Python	Main programming language
LangChain	LLM and tool integration
LangGraph	Agent orchestration
Groq	LLM and vision inference
ChromaDB	Vector database
Hugging Face	Text embeddings
DuckDuckGo	Web search
Wikipedia	General knowledge retrieval
Gradio	Web interface
PyPDF	PDF document processing
📁 Project Structure
week5-hybridsight/
│
├── agent.py
├── app.py
├── config.py
├── image_utils.py
├── ingest.py
├── tools_rag.py
├── tools_vision.py
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
File Description
agent.py

Contains the hybrid agent logic, LLM planning, tool execution, and final response synthesis.

app.py

Contains the Gradio application and connects the user interface with the HybridSight agent.

config.py

Contains model configuration, embedding configuration, ChromaDB settings, and environment variables.

ingest.py

Handles PDF loading, text splitting, metadata processing, and storing document embeddings in ChromaDB.

tools_rag.py

Implements the PDF retrieval tool used to search indexed documents.

tools_vision.py

Implements the image analysis tool using a Groq vision-capable model.

image_utils.py

Handles image conversion and preparation for vision processing.

⚙️ Installation
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/week5-hybridsight.git
cd week5-hybridsight
2. Create a virtual environment
python -m venv venv
3. Activate the virtual environment

On Windows PowerShell:

.\venv\Scripts\Activate.ps1
4. Install dependencies
pip install -r requirements.txt
🔑 Environment Variables

Create a .env file in the project root.

GROQ_API_KEY=your_groq_api_key

A template is provided in:

.env.example
Security

Never commit your .env file or API keys to GitHub.

The .gitignore file excludes .env from version control.

▶️ Running the Application

After activating the virtual environment, run:

python app.py

Gradio will provide a local URL in the terminal.

Open the URL in a browser to use HybridSight.

🧪 Example Questions
PDF / RAG

Upload a PDF and ask:

What is the main topic discussed in my PDF?
According to my uploaded document, explain the main concepts.
Vision

Upload an image and ask:

What is shown in this image?
Describe the main components of this image.
Web Search
What are the latest developments in artificial intelligence?
Wikipedia
Who invented the World Wide Web?
Hybrid Query

Upload a PDF and an image:

Compare the information shown in the image with the relevant information in my PDF.
🔄 How the System Works
Step 1 — PDF Upload

The user uploads a PDF through the Gradio interface.

Step 2 — Document Processing

The PDF is loaded and divided into smaller chunks.

Step 3 — Embeddings

The document chunks are converted into vector embeddings.

Step 4 — Vector Storage

The embeddings are stored in ChromaDB.

Step 5 — User Question

The user submits a natural-language question.

Step 6 — Tool Planning

The planner LLM analyzes the question and determines which information source is relevant.

Possible tools include:

search_documents
describe_image
web_search
wikipedia_search
Step 7 — Tool Execution

The selected tools are executed and their results are collected.

Step 8 — Response Synthesis

The synthesis LLM combines the retrieved information and generates the final answer.

Step 9 — Display

The final response is displayed in the Gradio chat interface.

🎯 Learning Objectives

This project demonstrates the following AI engineering concepts:

Retrieval-Augmented Generation (RAG)
Document loading and chunking
Text embeddings
Vector databases
Semantic search
LLM tool calling
AI agent architecture
LLM-based planning
Tool orchestration
Web search integration
Wikipedia integration
Vision-language models
Multi-source information processing
LangChain
LangGraph
Groq APIs
Gradio application development
🧠 Agent Workflow

HybridSight is designed around the following workflow:

User Question
     │
     ▼
Planner LLM
     │
     ├── PDF required? ──────► RAG Tool
     │
     ├── Image required? ────► Vision Tool
     │
     ├── Current information? ► Web Search
     │
     └── General knowledge? ──► Wikipedia
                                  │
                                  ▼
                            Tool Results
                                  │
                                  ▼
                           Synthesis LLM
                                  │
                                  ▼
                            Final Answer

The goal is to allow the LLM to determine the appropriate tool instead of relying on fixed keyword-based routing.

📌 Project Status

Completed — HybridSight AI Agent
Capability	Status
PDF Processing	✅
RAG	✅
ChromaDB	✅
Vision	✅
Web Search	✅
Wikipedia	✅
LLM Tool Planning	✅
Tool Execution	✅
Response Synthesis	✅
Gradio UI	✅
⚠️ Current Scope

The project focuses on learning and demonstrating the core architecture of a hybrid AI agent.

Individual capabilities such as PDF retrieval, image analysis, web search, and Wikipedia search are implemented and tested independently.

Complex multi-tool scenarios requiring simultaneous reasoning across multiple sources may require additional orchestration and optimization.

🔮 Possible Future Improvements
More robust multi-tool orchestration
Better document-level reasoning
Conversation memory
Streaming responses
Improved error handling
More vision capabilities
Source citations in final responses
Authentication and user management
Deployment to a cloud platform
Evaluation and automated agent testing
Better observability and tracing
👨‍💻 Author

Khush Hingrajiya

3rd Year Computer Engineering Student

Interested in:

Artificial Intelligence
Machine Learning
Generative AI
Data Structures & Algorithms
Software Development
📄 License

This project is created for educational and learning purposes.