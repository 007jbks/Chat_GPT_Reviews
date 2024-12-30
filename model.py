import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('ChatGPT_Reviews.csv')
data['Review'] = data['Review'].fillna("")
data['Ratings'] = data['Ratings'].fillna(3)

def rating_score(rating):
    if rating <= 2:
        return -1
    elif rating > 3:
        return 1
    else:
        return 0

data['Scores'] = data['Ratings'].apply(rating_score)
X = data['Review']
Y = data['Scores']

vectorizer = TfidfVectorizer(max_features=5000)
X_transformed = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []

for train_index, test_index in skf.split(X_transformed, Y):
    X_train, X_test = X_transformed[train_index], X_transformed[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    accuracy_scores.append(accuracy)
    print(f"Fold Accuracy: {accuracy}")
    print("Classification Report:\n", classification_report(Y_test, Y_pred))

average_accuracy = np.mean(accuracy_scores)
print(f"Average Accuracy across 5 folds: {average_accuracy}")

pos = np.sum(Y_pred == 1)
neu = np.sum(Y_pred == 0)
neg = np.sum(Y_pred == -1)
cat = ['Positive', 'Neutral', 'Negative']
val = [pos, neu, neg]
plt.bar(cat, val, color=['Green', 'Yellow', 'Red'])
plt.ylabel("No. of reviews")
plt.xlabel("Review Category")
plt.show()

review = input("Enter your review:")
review_transformed = vectorizer.transform([review])
result = model.predict(review_transformed)
if result[0] == 1:
    print("Positive")
elif result[0] == 0:
    print("Neutral")
else:
    print("Negative")
