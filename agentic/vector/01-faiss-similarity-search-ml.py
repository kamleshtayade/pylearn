import faiss
import numpy as np

# create a random datasets for vectors
data = np.random.rand(1000, 128).astype("float32")

# build the index
index = faiss.IndexFlatL2(128)  # create a flat (non-hierarchical) index - L2 distance index
index.add(data)  # add the vectors to the index

# query for nearest neighbors
query_vector = np.random.rand(1, 128).astype("float32")
k = 5
D, I = index.search(query_vector, k)  # search for 5 nearest neighbors of the query vector

print("Distances to nearest neighbors:", D)
print("Indices of nearest neighbors:", I)

##  python ./agentic/vector/01-faiss-similarity-search-ml.py