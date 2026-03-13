from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv
load_dotenv()

documents=[
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2-preview')

vector_store=Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory='./chroma_db',
    collection_name='contextual_retriever',
)

base_retriever = vector_store.as_retriever(
    search_kwargs={'k':2}
)
llm1=HuggingFaceEndpoint(
        repo_id='meta-llama/Llama-3.1-70B-Instruct',
        task='text-generation',
    )

comprssor = LLMChainExtractor.from_llm(
    llm=ChatHuggingFace(llm=llm1)
)

contextual_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=comprssor,
)

query = "What is photosynthesis?"

result = contextual_retriever.invoke(query)

for i,doc in enumerate(result):
    print(f"\n---Result({i+1})---\n")
    print(f"content:{doc.page_content}")


result2 = base_retriever.invoke(query)

for i,doc in enumerate(result2):
    print(f"\n---Result({i+1})---\n")
    print(f"content:{doc.page_content}")

