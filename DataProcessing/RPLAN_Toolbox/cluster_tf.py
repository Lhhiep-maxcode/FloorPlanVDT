import pickle
import numpy as np
import faiss

# Load precomputed data:
# - data.pkl contains the processed floorplan data (boxes, edges, etc.)
# - tf_discrete.pkl contains the turning functions for each floorplan (discretized as arrays)
data = pickle.load(open('./output/data.pkl','rb'))
tf_discrete = pickle.load(open('./output/tf_discrete.pkl','rb'))

# Build a numpy array of turning functions (TFs) for each sample in `data`
tf = np.array([tf_discrete[d['name']] for d in data]).astype(np.float32)
# Save as numpy array for later use
np.save('./output/tf_discrete.npy',tf)

# d = dimensionality of TF vectors
d = 1000
tf = tf.astype(np.float32)[:1000]   # Take first 1000 samples (probably for memory/training speed)

# ---- FAISS KMeans clustering ----
ncentroids = 1000   # number of clusters (k in k-means)
niter = 200         # number of iterations
verbose = True

# Initialize FAISS KMeans (GPU-accelerated)
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose,gpu=False)
# Train KMeans on the turning function vectors
kmeans.train(tf)
# Cluster centroids (the "average" TFs for each cluster)
centroids = kmeans.centroids

# ---- Nearest neighbor indexing ----
# Build a FAISS flat L2 index (brute-force nearest neighbor search)
index = faiss.IndexFlatL2(d)
# Add all TFs into the index
index.add(tf)
# For each centroid, find the nNN (=1000) nearest turning functions
nNN = 1000
D, I = index.search(kmeans.centroids, nNN)

# Save results:
# - centroids: the representative turning functions for each cluster
# - clusters (I): indices of TFs belonging to each centroid
np.save(f'./output/tf_centroids.npy',centroids)
np.save(f'./output/tf_clusters.npy',I)
