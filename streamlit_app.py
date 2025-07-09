import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Fungsi untuk ekstrak fitur dari gambar
def extract_features(image_path):
    try:
        image = Image.open(image_path).convert('RGB').resize((64, 64))
        return np.array(image).flatten()
    except:
        return None

# Judul Aplikasi
st.title("Unsupervised Image Clustering dengan Agglomerative")

# Ambil semua path gambar dari folder images/
image_folder = "images"
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")]

if len(image_paths) == 0:
    st.warning("Tidak ada gambar ditemukan di folder `images/`.")
    st.stop()

# Ekstraksi fitur
features = []
valid_image_paths = []

for path in image_paths:
    feat = extract_features(path)
    if feat is not None:
        features.append(feat)
        valid_image_paths.append(path)

features = np.array(features)

# Standardisasi
scaled_features = StandardScaler().fit_transform(features)

# PCA untuk reduksi dimensi (visualisasi)
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)

# Clustering
n_clusters = st.slider("Jumlah cluster", min_value=2, max_value=10, value=3)

model = AgglomerativeClustering(n_clusters=n_clusters)
labels = model.fit_predict(scaled_features)

score = silhouette_score(scaled_features, labels)
st.write(f"Silhouette Score: {score:.3f}")

# Visualisasi hasil clustering
fig, ax = plt.subplots()
scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='rainbow')
plt.title("Visualisasi Clustering")
st.pyplot(fig)

# Tampilkan gambar berdasarkan cluster
for cluster_num in range(n_clusters):
    st.markdown(f"### Cluster {cluster_num + 1}")
    cols = st.columns(5)
    idx = 0
    for i, path in enumerate(valid_image_paths):
        if labels[i] == cluster_num:
            with cols[idx % 5]:
                st.image(path, use_column_width=True)
            idx += 1
