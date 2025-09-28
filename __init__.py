# llm_execution_analysis.py

import os
import sys
import json
import logging
from dotenv import load_dotenv

# --- RAG/Vector DB Imports ---
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
# **FIXED IMPORTS (Addressing Deprecation Warnings):**
from langchain_chroma import Chroma  # NEW: Correct Chroma import
from langchain_huggingface import HuggingFaceEmbeddings  # NEW: Correct HuggingFace import
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Core LangChain Imports ---
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# --- Load Environment Variables ---
load_dotenv()

# --- Import Fix for Local Modules ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------------------------------------

from CodebaseAnalysis import CodebaseAnalysis
from config import REPO_PATH

# --- Configuration Constants ---
VECTOR_DB_PATH = "./chroma_db_code_analysis"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
TOP_K_CHUNKS = 5  # Number of relevant code chunks to retrieve

# --- LLM Setup & Authentication ---
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_KEY:
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_KEY:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not found.")

# **FIXED:** Using updated HuggingFaceEmbeddings import
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Using gemini-2.5-pro to maximize output capacity (best attempt to fix truncation)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=GEMINI_KEY,
    max_output_tokens=8192
)

# --- Structured Output & Parser ---
parser = JsonOutputParser(pydantic_object=CodebaseAnalysis)
format_instructions = parser.get_format_instructions()
repo_name = os.path.basename(REPO_PATH).replace("_Local", "")


# --- Code Loading and Indexing Function ---
def get_or_create_vector_db(repo_path: str, db_path: str, embeddings) -> Chroma:
    """Creates a Vector DB from code files if it doesn't exist, otherwise loads it."""

    # **FIXED:** Chroma import now refers to langchain_chroma.Chroma
    if os.path.exists(db_path):
        logging.info(f"Loading existing Vector DB from: {db_path}")
        return Chroma(persist_directory=db_path, embedding_function=embeddings)

    logging.info(f"Creating new Vector DB for project: {repo_path}")

    # 1. Define Code Splitter (Java-aware)
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    docs = []
    source_dir = os.path.join(repo_path, "src", "main", "java")

    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Java source directory not found at {source_dir}")

    # 2. Load and Split Documents
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    doc = Document(
                        content,
                        metadata={"source": file_path.replace(repo_path, ''), "project": repo_name}
                    )

                    chunks = text_splitter.split_documents([doc])
                    docs.extend(chunks)

                except Exception as e:
                    logging.warning(f"Could not process {file_path}. Error: {e}")

    if not docs:
        raise ValueError("No processable Java files found for indexing.")

    # 3. Create Vector DB and Persist
    logging.info(f"Creating {len(docs)} code chunks and saving to {db_path}")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=db_path
    )
    db.persist()
    return db


# --- Prompt Definition ---
SYSTEM_PROMPT = f"""
You are an expert Codebase Analyst. Your task is to analyze the provided Java code snippets from the '{repo_name}' project.
Use the provided code context to extract the project knowledge and format it strictly as a JSON object based on the required schema.
Focus on the main business logic files, NOT configuration, testing, or boilerplate code.
{{format_instructions}}
"""

HUMAN_PROMPT = """
Analyze the codebase and generate the structured knowledge based on the provided code context.
The project is: {repo_name}
Code Context: {context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT)
])


# --- Execution ---
def run_analysis():
    print(f"Running RAG chain for project: {repo_name}...")

    # --- 1. Get Vector Database and Retriever ---
    try:
        vectorstore = get_or_create_vector_db(REPO_PATH, VECTOR_DB_PATH, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_CHUNKS})
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ ERROR: Vector DB setup failed. {e}")
        return

    # --- 2. Build the RAG Chain ---
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # --- 3. Invoke the RAG Chain ---
    analysis_query = f"Perform a comprehensive code analysis and summary of the {repo_name} project's main functionalities, key methods, and architecture."

    try:
        response = retrieval_chain.invoke(
            {
                "input": analysis_query,
                "repo_name": repo_name,
                "format_instructions": format_instructions
            }
        )
    except Exception as e:
        print(f"\n❌ ERROR during chain execution: {e}")
        return

    # --- 4. Output Processing ---
    try:
        json_output_text = response.get('answer', '')
        json_output_dict = json.loads(json_output_text)

        # Structured Output (JSON File)
        output_json_path = f"{repo_name}_analysis_output.json"
        with open(output_json_path, 'w') as f:
            json.dump(json_output_dict, f, indent=4)

        print("-" * 50)
        print(f"✅ Analysis Complete. Structured JSON saved to: {output_json_path}")
        print("-" * 50)
        print("Extracted Knowledge Sample (Truncated):")
        print(json.dumps(json_output_dict, indent=4)[:1000] + "...")

    except json.JSONDecodeError:
        print("-" * 50)
        print("❌ ERROR: LLM did not return valid JSON.")
        print("Raw LLM Response (Truncated):")
        print(response.get('answer', 'Response key not found.')[:2000])
        print("-" * 50)


if __name__ == "__main__":
    run_analysis()