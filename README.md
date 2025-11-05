# 🎬 Movie Recommendation System (TMDB 5000 Dataset)

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow?logo=pandas)
![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-important?logo=google-colab)

---

## 📘 Project Overview

This project builds a **Movie Recommendation System** using data from **The Movie Database (TMDB 5000 Movies + Credits)**.  
It recommends movies similar to a given movie title by analyzing metadata like **genres**, **keywords**, **cast**, and **crew**.  

The model uses **Content-Based Filtering** — calculating **Cosine Similarity** on combined movie features.

---

## 🚀 Features

✅ Merges **two TMDB datasets** (`movies` + `credits`)  
✅ Extracts and cleans metadata (genres, cast, crew, keywords)  
✅ Uses **Bag-of-Words** (CountVectorizer) for feature representation  
✅ Computes **Cosine Similarity** for recommendation  
✅ Fully implemented in **Python** using **Pandas** and **Scikit-learn**  
✅ Runs smoothly on **Google Colab**  

---

## 🧩 Dataset

📦 **Dataset Source:** [TMDB 5000 Movie Dataset – Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

**Files Required:**
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Make sure to upload both files to your **Colab** working directory before running the notebook.

---

## 🧠 Methodology

1. **Data Preprocessing**  
   - Merge movie and credit datasets  
   - Extract genres, keywords, cast, and director  
   - Clean and normalize text data  

2. **Feature Engineering**  
   - Combine all textual info into a single column `tags`  
   - Apply `CountVectorizer` to create a word frequency matrix  

3. **Modeling**  
   - Compute pairwise **cosine similarity** between movie vectors  

4. **Recommendation Function**  
   - Retrieve top N most similar movies for a given input title  

---

## 🧮 Libraries Used

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
