
# 🎬 Movie Recommender System

This project demonstrates a **content-based movie recommender system** built using Python and scikit-learn. The system analyzes movie metadata to recommend similar movies based on user input.

---

## 📌 Project Objective

- Build a recommender system using movie metadata (overview, genre, cast, crew, etc.).
- Use text vectorization and cosine similarity to find related movies.
- Provide a user-friendly recommendation function based on movie titles.

---

## 📂 Dataset

The dataset was sourced from Kaggle and includes metadata on thousands of movies. Key fields used in this project include:

- **title**: The movie title.
- **overview**: A text description of the movie.
- **genres**: Thematic classification (e.g., Action, Comedy).
- **keywords**: Tags describing key elements in the movie.
- **cast** and **crew**: Names and roles of actors and filmmakers.

All text fields were preprocessed to remove null values and unify formatting before analysis.

---

## ⚙️ Methodology

### 1. Data Preprocessing

The key features (overview, keywords, cast, crew, genres) were cleaned and merged into a single `tags` column for similarity comparison.

```python
# Preprocessing steps
movies.dropna(inplace=True)
movies['tags'] = movies['overview'] + ' ' + movies['keywords'] + ' ' + movies['cast'] + ' ' + movies['crew']
```

---

### 2. Text Vectorization with CountVectorizer

We used **CountVectorizer** to convert text into numerical vectors, which enables comparison between movies.

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
```

---

### 3. Cosine Similarity

Cosine similarity was used to calculate similarity between movies based on their vector representations.

```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
```

---

### 4. Recommendation Function

A simple function was defined to take a movie title and return the top 5 most similar movies.

```python
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in movies_list:
        print(movies.iloc[i[0]].title)
```

---

## ✅ Example Output

```python
recommend('The Dark Knight')
```

**Output:**
```
Batman Begins
The Dark Knight Rises
Batman v Superman: Dawn of Justice
Man of Steel
Suicide Squad
```

---

## 🧠 Conclusion

This project highlights the strength of content-based filtering for media recommendations. By leveraging metadata and cosine similarity, we can create lightweight, interpretable recommender systems without deep user behavior tracking.

📌 **Next Steps**:
- Enhance text cleaning and feature engineering.
- Incorporate TF-IDF weighting.
- Add a web interface using Streamlit or Flask.

---

## 🚀 Run This Project on Binder

Click below to interact with the notebook directly in your browser:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/itsgoni1/Portfolio/main?filepath=Movie_final.ipynb)

---
