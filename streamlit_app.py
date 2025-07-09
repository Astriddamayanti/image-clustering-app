# streamlit_app.py

import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image as PILImage
from tqdm import tqdm
import cv2

st.set_page_config(layout="wide")
st.title("📊 Unsupervised Image Clustering (KMeans, Hierarchical, DBSCAN)")

uploaded_files = st.file_uploader(
    "📤 Upload beberapa gambar (.jpg, .png):", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    st.subheader("1. Ekstraksi Fitur Gambar dengan ResNet50")
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)

    def extract_features_from_file(uploaded_file):
        img = PILImage.open(uploaded_file).convert("RGB").resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array, verbose=0)
        return features.flatten()

    features = np.array([extract_features_from_file(f) for f in tqdm(uploaded_files)])
    st.write("Jumlah gambar:", len(uploaded_files))
    st.write("Dimensi fitur:", features.shape)

    # PCA untuk visualisasi
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)

    st.subheader("2. KMeans Clustering")
    n_clusters = st.slider("Pilih jumlah cluster (K):", 2, 10, 4)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(features)
    sil_score = silhouette_score(features, kmeans_labels)
    st.write(f"Silhouette Score (KMeans): {sil_score:.3f}")
    fig1, ax1 = plt.subplots()
    ax1.scatter(reduced[:, 0], reduced[:, 1], c=kmeans_labels, cmap='tab10')
    ax1.set_title("KMeans PCA Visualization")
    st.pyplot(fig1)

    st.subheader("3. Hierarchical Clustering (Agglomerative)")
    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    agglo_labels = agglo.fit_predict(features)
    fig2, ax2 = plt.subplots()
    ax2.scatter(reduced[:, 0], reduced[:, 1], c=agglo_labels, cmap='tab10')
    ax2.set_title("Hierarchical PCA Visualization")
    st.pyplot(fig2)

    st.subheader("4. DBSCAN Clustering")
    eps = st.slider("Nilai epsilon (eps) DBSCAN", 5, 50, 30)
    dbscan = DBSCAN(eps=eps, min_samples=3)
    dbscan_labels = dbscan.fit_predict(features)
    fig3, ax3 = plt.subplots()
    ax3.scatter(reduced[:, 0], reduced[:, 1], c=dbscan_labels, cmap='tab10')
    ax3.set_title("DBSCAN PCA Visualization")
    st.pyplot(fig3)

    if len(set(dbscan_labels)) > 1:
        sil_dbscan = silhouette_score(features, dbscan_labels)
        st.write(f"Silhouette Score (DBSCAN): {sil_dbscan:.3f}")
    else:
        st.warning("DBSCAN menghasilkan hanya satu cluster atau semua noise.")
else:
    st.info("⬆️ Silakan upload beberapa gambar terlebih dahulu.")
