from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

EMBEDDING_MODEL="mxbai-embed-large"
PERSIST_DIRECTORY="db/chroma"

def get_relevant_docs():
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

    return retriever

def refine_output_with_llm(chat_history, query, retriever):
    llm = ChatOllama(
        model="qwen:7b",
        temperature=0
    )
    
    if len(chat_history) > 0:
        messages = [
            ("system", "Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
            ("human", f"Last Question: {query}, Chat History: {chat_history}")
        ]

        result = llm.invoke(messages)
        
        user_messages = result.content.strip()

    else:
        user_messages = query

    chat_history.append(query)

    relevant_docs = retriever.invoke(
        query
    )

    print(f"User Query: {user_messages}")
    print("-" * 100)

    messages = [
        ("system", "You're a assistant and you'll help me get the best answer out of five or less docs with refined output to user only if the answer is accurate and available in the docs with respect to the query that you'll be provided. If the relevant docs don't have matching answer to the query just say that you cannot assist with queries out of the scope of the system"),
        ("human", f"Documents: {relevant_docs} \n Query with history (if applicable): {user_messages}")
    ]
    
    
    return llm.invoke(messages) 

def main():
    retriever = get_relevant_docs()
    chat_history = []

    while True:
        query=input("What do you want to known: ") 

        #check if there's a command
        if query == "/quit":
            return 1
        elif query.startswith("/"):
            return "Invalid command provided."
        
        print("\n" + "-" * 100)
                    
        response = refine_output_with_llm(
            chat_history=chat_history,
            query=query,
            retriever=retriever
        ) 

        print(response.content)
        print("\n" + "-" * 100 + "\n")

if __name__=="__main__":
    main()
