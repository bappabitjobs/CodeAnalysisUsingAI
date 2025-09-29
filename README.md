 Codebase Analysis Project
 This project provides a robust, cost-effective solution for performing static codebase analysis using Generative AI (GenAI). It employs a Retrieval-Augmented Generation (RAG) architecture with completely local and free components for the knowledge indexing phase, ensuring analysis can be performed without hitting API rate limits for embeddings.
  Features
RAG Architecture: Uses an efficient RAG pipeline to pull relevant code snippets for deep analysis.

Cost-Efficient Embeddings: Leverages the free, local HuggingFace all-MiniLM-L6-v2 model for generating embeddings, completely bypassing Google API quota limits for this task.

Local Vector Database: Uses ChromaDB as a fast, file-based vector store to persist the code knowledge, eliminating the need for external cloud database services.

Structured Output: Generates a structured JSON output based on a Pydantic schema (CodebaseAnalysis.py) for easy consumption and integration.

High-Capacity LLM: Utilizes the Gemini 2.5 Pro model for the final generation step, maximizing the chance of complete, high-quality analysis.


Prerequisites
1. Python 3.9+

2. A Gemini API Key: Set as an environment variable (see Setup).

3. Target Codebase: Your Java project files must be located at the path defined in config.py (e.g., E:\Code\).


Setup and Installation
1. Create and Activate Virtual Environment
python -m venv .venv
# On Windows
.\.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate


2. Install Dependencies
The project uses the latest, non-deprecated LangChain packages:
pip install -U langchain-google-genai langchain-huggingface langchain-chroma \
    langchain-text-splitters chromadb sentence-transformers pydantic python-dotenv

3. Set Up Environment Variable
Create a file named .env in the root directory of this project (E:\Code\CodeAnalysisProject\) and add your Gemini API key:
# .env file
# Get your key from Google AI Studio
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"



4. Configure Target Project
Ensure your config.py file points to the root directory of the Java codebase you wish to analyze:
# config.py example
REPO_PATH = "E:/Code/*"


Usage
Run the main analysis script from your project root. The script will automatically handle the creation or loading of the Vector Database.
python __init__.py

Output Flow
1. Vector DB Creation: On the first run, the script will load all Java files, split them into chunks, generate embeddings using the local HuggingFace model, and save the database to ./chroma_db_code_analysis.
2. RAG Execution: The script performs a retrieval query to fetch relevant code chunks.
3. Analysis Generation: The LLM generates the final, structured JSON output.
4. Result: The structured output will be saved to a file named *SProject_analysis_output.json in the project root.


Architecture Details
   1.1 Component: Document Loader , Loads Java source files.	

   1.2 Role : Loads Java source files.

   1.3 Technology Used: os.walk, file I/O


   2.1 Text Splitter: 
   2.2 Role : Breaks large files into context-preserving chunks.
   2.3 Technology Used: Preserves structure for RAG.

   3.1 Embeddings :
   3.2 Role : Converts code chunks into vector space.
   3.3 Technology Used: HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

   4.1 Vector DB:
   4.2 Role : Stores the code vectors for semantic search.
   4.3 Technology Used : Chroma

    5.1 LLM :
    5.2 Role : Performs the final knowledge structuring and analysis.
    5.3 Technology Used : ChatGoogleGenerativeAI(model="gemini-2.5-pro")

    6.1 Output Schema : 
    6.2 Role : Defines the mandatory structure for the result.
    6.3 Technology Used : LlamaIndex


