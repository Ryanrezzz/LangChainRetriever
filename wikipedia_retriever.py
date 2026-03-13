from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2,lang='en')

query = "When did India win the first 50 overs World Cup?"

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...") 

