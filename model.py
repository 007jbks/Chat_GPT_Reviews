import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('ChatGPT_Reviews.csv')
data['Review'] = data['Review'].fillna("")
data['Ratings'] = data['Ratings'].fillna(3)


def rating_score(rating):
	if rating<=2:
		return -1
	elif rating>3:
		return 1
	else:
		return 0
data['Scores']  = data['Ratings'].apply(rating_score)
X = data['Review']
Y = data['Scores']
X_test,X_train,Y_test,Y_train = train_test_split(X,Y,test_size=0.4,random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

model = LogisticRegression()

model.fit(X_train_transformed,Y_train)
Y_pred = model.predict(X_test_transformed)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))


#Overall Data:

#After Application of model: Test Data
pos = np.sum(Y_pred==1)
neu = np.sum(Y_pred==0)
neg = np.sum(Y_pred==-1)
cat = ['Positive','Neutral','Negative']
val = [pos,neu,neg]
plt.bar(cat,val,color=['Green','Yellow','Red'])
plt.ylabel("No. of reviews")
plt.xlabel("Review Category")
plt.show()

review = input("Enter your review:")
review_transformed = vectorizer.transform([review])
result = model.predict(review_transformed)
if result[0]==1:
	print("Positive")
elif result[0]==0:
	print("Neutral")
else:
	print("Negative")	

