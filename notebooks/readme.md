# Notebooks: NLP Pipeline and Model Development

This folder contains the notebooks used to build and analyze the NLP pipeline behind the **Scientific Clusters Explorer** application.

These notebooks document the full workflow, from raw data processing to clustering and model preparation for real-time inference.

---

## Contents

- **01_nlp_clustering_pipeline.ipynb**  
  Full end-to-end pipeline for text preprocessing, TF-IDF feature extraction, and KMeans clustering.

- **02_live_text_classifier_training.ipynb**  
  Training of a supervised classifier using cluster assignments as pseudo-labels, enabling real-time text classification in the Streamlit app.

---

## What these notebooks demonstrate

- Text preprocessing and normalization of scientific abstracts  
- Feature engineering using TF-IDF (unigrams and bigrams)  
- Unsupervised learning with KMeans for topic discovery  
- Cluster interpretation through top terms and representative documents  
- Transition from unsupervised to supervised learning (pseudo-labeling)  
- Training a lightweight classifier for real-time inference  

---

## Dataset

The project uses the **CORD-19 (COVID-19 Open Research Dataset)**, which contains a large collection of scientific articles related to COVID-19 and related topics.

You can download the dataset from Kaggle: [Download the dataset](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge?resource=download)


---

## Notes

- The dataset is not included in this repository due to its size  
- Intermediate processed files (e.g., `clustered_docs.csv`) are used by the Streamlit app  
- The notebooks were designed for experimentation and development, while the app uses precomputed outputs for efficiency  

---

## How this connects to the app

These notebooks generate the artifacts used in the Streamlit application:

- `clustered_docs.csv` → clustered documents  
- `cluster_stats.csv` → cluster-level metrics  
- `tfidf_vectorizer.joblib` → text vectorization model  
- `cluster_classifier.joblib` → trained classifier for live predictions  

---

## Purpose

The goal of this folder is to show the full data science workflow behind the application:

from raw text → to NLP processing → to clustering → to real-time prediction

This complements the Streamlit app by providing transparency, reproducibility, and technical depth.

---
