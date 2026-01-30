from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os

EMBEDDING_MODEL="mxbai-embed-large"
PERSIST_DIRECTORY="db/chroma"

def load_documents(docs_dir="docs"):
    if not os.path.exists("docs"):
        raise FileNotFoundError("Invalid direcotry provided")

    loader = DirectoryLoader(
        path=docs_dir,
        glob="*.txt",
        loader_cls=TextLoader
    )
    
    documents = loader.load()

    if len(documents) == 0:
        print("No documents found to index")
    
    if documents:
        print(f" Total Docs: {len(documents)} \n")
        for i,doc in enumerate(documents):
            print(f"Document {i+1}: ")
            print(f"    Page Content: {doc.page_content[:100]}")
            print(f"    Page Source: {doc.metadata['source']} \n")

    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=0):
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks[:5]):
        print(f"Chunk: {i+1}")
        print(f"    Data: {chunk.page_content}")
        print("f    Metadata: {chunk.metadata}")
        print("")
        print("-" * 50)

        if len(chunks) > 5:
            print(f" and .. {len(chunks) - 5 } more chunks")
            print("-" * 50)

    return chunks

def create_vectors(chunks):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vectors = Chroma.from_documents(
        persist_directory=PERSIST_DIRECTORY,
        documents=chunks,
        embedding=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )

    return vectors

def main():
    documents = load_documents(docs_dir="docs")
    chunks = chunk_documents(documents)
    return create_vectors(chunks)

if __name__ == "__main__":
    main()
