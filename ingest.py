import os

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone


load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)
index = pc.Index("document-search")


with open("sample.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

chunks = [line.strip() for line in lines if line.strip()]


print("Number of chunks:", len(chunks))

for chunk_index, chunk in enumerate(chunks):
    print(chunk_index, chunk)


model = SentenceTransformer(
     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

embeddings = model.encode(chunks)

records = []
for chunk_index, chunk in enumerate(chunks):
    records.append(
        {
            "id":f"chunk-{chunk_index}",
            "values": embeddings[chunk_index].tolist(),
            "metadata": {
                "source": "sample.txt",
                "text" : chunk,
                "chunk_id": chunk_index,
            },
        }
    )
print ("Records ready:", len(records))

response = index.upsert(vectors=records)

print("Upsert response:", response)
print(index.describe_index_stats())

query = "What is Pinecine used for ?"

query_embedding = model.encode(query).tolist()

results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True,
)

print("search results:")

for match in results["matches"]:
    print("Score:", match["score"])
    print("Text:", match["metadata"]["text"])
    print("Source:", match["metadata"]["source"])
    print("---")