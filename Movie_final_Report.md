```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random

```


```python
file_path = r"C:\Users\itsgo\Documents\Movie Reccomender\Expanded_Movie_Recommender_Data__200__Movies_.csv"
df = pd.read_csv(file_path)
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age Group</th>
      <th>Gender</th>
      <th>Occupation</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children's</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>...</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
      <th>Top Movie 1</th>
      <th>Top Movie 2</th>
      <th>Top Movie 3</th>
      <th>Top Movie 4</th>
      <th>Top Movie 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56+</td>
      <td>Female</td>
      <td>College/Grad Student</td>
      <td>Like</td>
      <td>Like</td>
      <td>Neutral</td>
      <td>Like</td>
      <td>Like</td>
      <td>Like</td>
      <td>Dislike</td>
      <td>...</td>
      <td>Like</td>
      <td>Like</td>
      <td>Like</td>
      <td>Dislike</td>
      <td>Dislike</td>
      <td>The Ring</td>
      <td>Inception</td>
      <td>Elysium</td>
      <td>Gone Girl</td>
      <td>Pride and Prejudice</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50-55</td>
      <td>Female</td>
      <td>Scientist</td>
      <td>Neutral</td>
      <td>Neutral</td>
      <td>Like</td>
      <td>Neutral</td>
      <td>Like</td>
      <td>Like</td>
      <td>Like</td>
      <td>...</td>
      <td>Neutral</td>
      <td>Like</td>
      <td>Dislike</td>
      <td>Like</td>
      <td>Like</td>
      <td>Pulp Fiction</td>
      <td>Finding Nemo</td>
      <td>The Godfather</td>
      <td>Seven</td>
      <td>No Country for Old Men</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25-34</td>
      <td>Female</td>
      <td>Other</td>
      <td>Dislike</td>
      <td>Like</td>
      <td>Like</td>
      <td>Neutral</td>
      <td>Like</td>
      <td>Dislike</td>
      <td>Like</td>
      <td>...</td>
      <td>Neutral</td>
      <td>Dislike</td>
      <td>Like</td>
      <td>Like</td>
      <td>Like</td>
      <td>Gone Girl</td>
      <td>Get Out</td>
      <td>Free Solo</td>
      <td>Bridesmaids</td>
      <td>Touch of Evil</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25-34</td>
      <td>Female</td>
      <td>Tradesman/Craftsman</td>
      <td>Like</td>
      <td>Neutral</td>
      <td>Neutral</td>
      <td>Like</td>
      <td>Like</td>
      <td>Like</td>
      <td>Like</td>
      <td>...</td>
      <td>Like</td>
      <td>Like</td>
      <td>Neutral</td>
      <td>Like</td>
      <td>Like</td>
      <td>Brave</td>
      <td>Free Solo</td>
      <td>The Avengers</td>
      <td>The Bourne Ultimatum</td>
      <td>Mean Girls</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25-34</td>
      <td>Female</td>
      <td>Farmer</td>
      <td>Neutral</td>
      <td>Dislike</td>
      <td>Like</td>
      <td>Dislike</td>
      <td>Like</td>
      <td>Neutral</td>
      <td>Like</td>
      <td>...</td>
      <td>Neutral</td>
      <td>Neutral</td>
      <td>Like</td>
      <td>Like</td>
      <td>Dislike</td>
      <td>Dunkirk</td>
      <td>Singin' in the Rain</td>
      <td>Inside Job</td>
      <td>Prisoners</td>
      <td>The Sound of Music</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
preference_map = {"Like": 1, "Neutral": 0, "Dislike": -1}
genre_columns = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

df_numeric = df.copy()
for genre in genre_columns:
    df_numeric[genre] = df_numeric[genre].map(preference_map)

```


```python
label_encoders = {}
for col in ['Gender', 'Age Group', 'Occupation']:
    le = LabelEncoder()
    df_numeric[col] = le.fit_transform(df[col])
    label_encoders[col] = le

```


```python
all_features = genre_columns + ['Gender', 'Age Group', 'Occupation']
scaler = StandardScaler()
user_features_scaled = scaler.fit_transform(df_numeric[all_features])

```


```python
age_groups = [
    "Under 18", "18-24", "25-34", "35-44", 
    "45-49", "50-55", "56+"
]

occupations = [
    "Other", "Academic/Educator", "Artist", "Clerical/Admin", 
    "College/Grad Student", "Customer Service", "Doctor/Health Care", 
    "Executive/Managerial", "Farmer", "Homemaker", "K-12 Student", 
    "Lawyer", "Programmer", "Retired", "Sales/Marketing", 
    "Scientist", "Self-Employed", "Technician/Engineer", 
    "Tradesman/Craftsman", "Unemployed", "Writer"
]

```


```python
def ask_user_preferences():
    print("Please rate your interest in each genre as: Like / Neutral / Dislike\n")
    user_prefs = {}
    for genre in genre_columns:
        while True:
            ans = input(f"{genre}: ").strip().capitalize()
            if ans in preference_map:
                user_prefs[genre] = preference_map[ans]
                break
            else:
                print("Invalid input. Please type Like, Neutral, or Dislike.")
    return user_prefs

```


```python
def get_user_vector():
    prefs = ask_user_preferences()

    # Gender input loop
    while True:
        gender = input("\nGender (Male/Female/Other): ").strip().capitalize()
        if gender in label_encoders['Gender'].classes_:
            gender_enc = label_encoders['Gender'].transform([gender])[0]
            break
        else:
            print("❌ Invalid gender. Please enter one of:", list(label_encoders['Gender'].classes_))

    # Age group input loop
    print("\nChoose your Age Group from the following:")
    for ag in label_encoders['Age Group'].classes_:
        print(f"- {ag}")
    while True:
        age = input("Age Group: ").strip()
        if age in label_encoders['Age Group'].classes_:
            age_enc = label_encoders['Age Group'].transform([age])[0]
            break
        else:
            print("❌ Invalid age group. Please enter one exactly as shown.")

    # Occupation input loop
    print("\nChoose your Occupation from the following:")
    for job in label_encoders['Occupation'].classes_:
        print(f"- {job}")
    while True:
        occ = input("Occupation: ").strip()
        if occ in label_encoders['Occupation'].classes_:
            occ_enc = label_encoders['Occupation'].transform([occ])[0]
            break
        else:
            print("❌ Invalid occupation. Please enter one exactly as shown.")

    full_vector = [prefs[genre] for genre in genre_columns] + [gender_enc, age_enc, occ_enc]
    return scaler.transform([full_vector])[0]

```


```python
def recommend_by_user_similarity(user_vector, df_source, top_n=3):
    similarities = cosine_similarity([user_vector], user_features_scaled)[0]
    top_indices = similarities.argsort()[::-1][1:11]  # top 10 similar users
    top_movies = df_source.iloc[top_indices][[f"Top Movie {i+1}" for i in range(5)]].values.ravel()
    top_movies = pd.Series(top_movies).value_counts().head(top_n)
    return top_movies.index.tolist()

```


```python
def hybrid_recommender():
    user_vector = get_user_vector()
    if user_vector is not None:
        recommendations = recommend_by_user_similarity(user_vector, df)
        print("\n🎬 Based on your preferences and demographics, we recommend:")
        for i, movie in enumerate(recommendations, 1):
            print(f"{i}. {movie}")

```


```python
hybrid_recommender()

```
