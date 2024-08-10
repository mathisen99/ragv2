from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load API key from environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Constants for configuration
K_RESULTS = 3  # Number of results to retrieve
SIMILARITY_THRESHOLD = 0.5  # Similarity threshold for search results

def ask_question(query):
    # Load embeddings and Chroma vector store
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # Use the correct model for chat completion
    llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o")

    # Create the retriever with specified parameters
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": K_RESULTS, "score_threshold": SIMILARITY_THRESHOLD}
    )

    # Create the retrieval QA chain
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    # Query the chain and get the response
    response = chain.invoke({"query": query})
    
    # Check if source documents are available in the response
    if 'source_documents' in response:
        for doc in response['source_documents']:
            print(f"Source Document: {doc.metadata['source']}, Section: {doc.metadata.get('section', 'N/A')}\nContent: {doc.page_content}\n")

    return response.get("result", "No result found.")

if __name__ == "__main__":
    query = "i need to build a website and i need tts features" 
    print(ask_question(query))
