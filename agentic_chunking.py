from langchain_ollama import ChatOllama
import ast
from pydantic import BaseModel, Field
from typing import List

class StructuredResponse(BaseModel):
    chunk: List[str] = Field(description="The list of logical text chunks")

llm = ChatOllama(model="qwen2.5:14b")

tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

prompt = f"""
You're a text chunking expert for RAG systems. Take the given text and convert it into chunks.
You're to make sure the chunks are logical and each chunk doesn't loose meaning in any way.

Rules:
- Each chunk should be around 200 character.
- Split at natural topic boundaries.
= Keep related information together

Text:
{tesla_text}
"""

structured_llm = llm.with_structured_output(StructuredResponse)
response = structured_llm.invoke(prompt)

for i,chunk in enumerate(response.chunk):
    print(f"Chunk {i+1}: {chunk} \n")
