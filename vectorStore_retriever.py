from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

vector_store= Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory="./chroma_db",
    collection_name="langchain_collection"
)

retriever= vector_store.as_retriever(search_kwargs={"k": 2})
query ="What is chroma?"

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")

