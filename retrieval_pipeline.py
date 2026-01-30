from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

EMBEDDING_MODEL="mxbai-embed-large"
PERSIST_DIRECTORY="db/chroma"

def main():
    query=input("What do you want to known: ") 
    embeddings=OllamaEmbeddings(model=EMBEDDING_MODEL) 

    chroma = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_metadata={
            "hnsw:space": "cosine"
        }
    )

    retriever = chroma.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.5
        },
    )

    relevant_docs = retriever.invoke(query)

    for i, doc in enumerate(relevant_docs):
        print(f"Document: {i+1}")
        print(f"    Chunk: {doc}")
        print("-" * 50 + "\n")

if __name__=="__main__":
    main()
