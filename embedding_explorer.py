from sentence_transformers import SentenceTransformer
from dataset import sentences
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

embeddings = model.encode(sentences)


model_b = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings_b = model_b.encode(sentences)

print("Model B shape:", embeddings_b.shape)
print("Model B vector dimension:", embeddings_b.shape[1])

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
)

points_2d = tsne.fit_transform(embeddings)
print("t-SNE shape:", points_2d.shape)

print("Embedding shape:", embeddings.shape)
print("Number of embeddings:", embeddings.shape[0])
print("Vector dimension:", embeddings.shape[1])


def cosine_similarity(vector_a, vector_b):
    dot_product = vector_a @ vector_b
    norm_a = (vector_a @ vector_a) ** 0.5
    norm_b = (vector_b @ vector_b) ** 0.5

    return dot_product / (norm_a * norm_b)

query = "What sport improves cardiovascular fitness?"
query_embedding = model.encode(query)

results = []

for index, sentence_embedding in enumerate(embeddings):
    score = cosine_similarity(query_embedding, sentence_embedding)

    results.append((score, sentences[index]))

results.sort(reverse=True)

print("Model A Top 5:")
print(results[:5])

labels = (
    ["Programming"] * 20
    + ["AI"] * 20
    + ["Sports"] * 20
    + ["Food"] * 20
    + ["Travel"] * 20
)

plt.figure(figsize=(10, 7))

for label in set(labels):
    indices = [i for i, item in enumerate(labels) if item == label]

    x = points_2d[indices, 0]
    y = points_2d[indices, 1]

    plt.scatter(x, y, label=label)

plt.legend()
plt.title("t-SNE Visualization of Sentence Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.savefig("tsne_embeddings.png")

query_embedding_b = model_b.encode(query)

results_b = []

for index, sentence_embedding in enumerate(embeddings_b):
    score = cosine_similarity(query_embedding_b, sentence_embedding)
    results_b.append((score, sentences[index]))

results_b.sort(reverse=True)

print("Model B Top 5:")
print(results_b[:5])