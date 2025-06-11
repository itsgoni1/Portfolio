import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random


preference_map = {"Like": 1, "Neutral": 0, "Dislike": -1}
genre_columns = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

df_numeric = df.copy()
for genre in genre_columns:
    df_numeric[genre] = df_numeric[genre].map(preference_map)


label_encoders = {}
for col in ['Gender', 'Age Group', 'Occupation']:
    le = LabelEncoder()
    df_numeric[col] = le.fit_transform(df[col])
    label_encoders[col] = le


all_features = genre_columns + ['Gender', 'Age Group', 'Occupation']
scaler = StandardScaler()
user_features_scaled = scaler.fit_transform(df_numeric[all_features])


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


def recommend_by_user_similarity(user_vector, df_source, top_n=3):
    similarities = cosine_similarity([user_vector], user_features_scaled)[0]
    top_indices = similarities.argsort()[::-1][1:11]  # top 10 similar users
    top_movies = df_source.iloc[top_indices][[f"Top Movie {i+1}" for i in range(5)]].values.ravel()
    top_movies = pd.Series(top_movies).value_counts().head(top_n)
    return top_movies.index.tolist()


def hybrid_recommender():
    user_vector = get_user_vector()
    if user_vector is not None:
        recommendations = recommend_by_user_similarity(user_vector, df)
        print("\n🎬 Based on your preferences and demographics, we recommend:")
        for i, movie in enumerate(recommendations, 1):
            print(f"{i}. {movie}")


hybrid_recommender()
