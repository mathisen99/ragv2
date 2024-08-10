from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv

# Load API key from environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def ingest_docs(data_folder_path):
    # Load and split the text files into chunks
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    
    for filename in os.listdir(data_folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                docs = [Document(page_content=text)]
                chunks.extend(text_splitter.split_documents(docs))
    
    # Create embeddings using OpenAI and store in ChromaDB
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")

    return vector_store

if __name__ == "__main__":
    ingest_docs("./data")
