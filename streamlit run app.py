# app.py
import streamlit as st
import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Fungsi bantu untuk load gambar dari folder
def load_images(image_folder):
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder)
                   if fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.jpeg')]
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64))
        images.append(img)
    return image_paths, np.array(images)

# Ekstraksi fitur sederhana (rata-rata warna RGB)
def extract_features(images):
    features = []
    for img in images:
        mean_color = img.mean(axis=(0, 1))
        features.append(mean_color)
    return np.array(features)

# Streamlit UI
st.title("Aplikasi Clustering Gambar dengan K-Means")

uploaded_folder = st.text_input("Masukkan path folder gambar:", "images")

if os.path.exists(uploaded_folder):
    image_paths, images = load_images(uploaded_folder)
    features = extract_features(images)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    n_clusters = st.slider("Jumlah cluster", min_value=2, max_value=10, value=3)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_features)

    st.write(f"### Hasil Clustering menjadi {n_clusters} cluster")

    for cluster_id in range(n_clusters):
        st.write(f"#### Cluster {cluster_id + 1}")
        cluster_indices = np.where(labels == cluster_id)[0][:3]  # tampilkan 3 gambar saja
        cols = st.columns(3)
        for i, idx in enumerate(cluster_indices):
            img = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
            cols[i].image(img, caption=f"Gambar {idx + 1}", use_column_width=True)
else:
    st.warning("Folder tidak ditemukan. Pastikan path-nya benar dan folder berisi gambar .jpg/.png.")

