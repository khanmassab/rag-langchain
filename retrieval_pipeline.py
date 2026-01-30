from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

EMBEDDING_MODEL="mxbai-embed-large"
PERSIST_DIRECTORY="db/chroma"

def get_relevant_docs(query):
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
            "score_threshold": 0.3
        },
    )

    relevant_docs = retriever.invoke(query)

    return relevant_docs 

def refine_output_with_llm(relevant_docs, query):
    
    llm = ChatOllama(
        model="qwen:7b",
        temperature=0
    )

    messages = [
        ("system", "You're a assistant and you'll help me get the best answer out of five or less docs with refined output to user only if the answer is accurate and available in the docs with respect to the query that you'll be provided. If the relevant docs don't have matching answer to the query just say that you cannot assist with queries out of the scope of the system"),
        ("human", f"Documents: {relevant_docs} \n Query with history (if applicable): {query}")
    ]
    
    return llm.invoke(messages) 

def main():
    history = []
    
    while True:
        query=input("What do you want to known: ") 
    
        #check if there's a command
        if query == "/quit":
            return 1
        elif query[:1] == "/":
            return "Invalid command provided."
    

        if history and len(history) > 0:
            query.append = ("prompt_history", f"You're to remember the prompt history in this chat and answer based on the following context: {history}")
    
        relevant_docs = get_relevant_docs(query=query)
   
        response = refine_output_with_llm(
            relevant_docs=relevant_docs,                      
            query=query
        ) 

        print(response.content)
    
    print("Thank you for using this chatbot")
if __name__=="__main__":
    main()
