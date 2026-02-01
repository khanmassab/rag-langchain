from langchain_text_splitters import CharacterTextSplitter 

spiltter = CharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separator="\n"
    #separators=['\n\n', '\n', '. ', ', ']
)

text = """LangChain is a framework for developing applications.
It provides tools for working with LLMs.
Text splitting is an important preprocessing step.
It helps manage large documents efficiently.
Chunks are then passed to embedding models."""

chunks = spiltter.split_text(
    text
)


print()

for i,chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk} \n")
    print(f"Length: {len(chunk)} \n")
    print("-" * 100)


print()
