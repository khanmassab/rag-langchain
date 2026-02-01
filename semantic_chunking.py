from langchain_experimental.text_splitter import SemanticChunker 
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

chunker = SemanticChunker(
    embeddings = embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70
)

tesla_text = """Tesla's Q3 Results:
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance:
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges:
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

chunks = chunker.split_text(tesla_text)

print("\n ")
for i,chunk in enumerate(chunks):
    print(f"Chunk No: {i+1} \nLength: {len(chunk)}\nContent: \n{chunk} \n")
    print("-" * 100)
