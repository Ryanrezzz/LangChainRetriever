import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

from langchain_core.documents import Document


documents=[
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]
embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2-preview')
vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embedding,

)

retriever= vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={
        'k':2,
        'lambda_mult':0.5
    }
)
query='What is LangChain?'
result = retriever.invoke(query)
for i,doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")
