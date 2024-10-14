import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# This is the Dataset for webtoon 
webtoon_data = [
    ("A high school girl is caught between two boys as she learns about beauty and love.", "Romance"),
    ("A group of heroes band together to save their world from destruction.", "Action"),
    ("A young boy embarks on a journey in a magical world full of strange creatures.", "Fantasy"),
    ("Two unlikely individuals fall in love while overcoming their personal struggles.", "Romance"),
    ("A warrior seeks revenge after his family is killed by a powerful enemy.", "Action"),
    ("A girl discovers she has special powers and must fight dark forces.", "Fantasy"),
    ("A romantic comedy set in a high school where students compete in dating games.", "Romance"),
    ("A legendary hero returns to fight against an invading army.", "Action"),
    ("A young princess must navigate political intrigue and magic to reclaim her kingdom.", "Fantasy"),
    ("Two childhood friends reunite and find themselves falling in love again.", "Romance"),
    ("A secret agent goes on a dangerous mission to stop a criminal organization.", "Action"),
    ("A young witch learns about her magical heritage and must save the world.", "Fantasy")
]

webtoon_descriptions = [desc for desc, genre in webtoon_data]
webtoon_genres = [genre for desc, genre in webtoon_data]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(webtoon_descriptions)
y = webtoon_genres


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:\n", classification_report(y_test, y_pred))


new_description = ["A boy who can control fire must protect his kingdom from an evil sorcerer."]
new_description_tfidf = vectorizer.transform(new_description)
predicted_genre = classifier.predict(new_description_tfidf)
print(f"Predicted genre: {predicted_genre[0]}")
