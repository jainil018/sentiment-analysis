import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import joblib

# Load dataset
df = pd.read_csv("YoutubeCommentsDataSet.csv")

# Drop missing values
df.dropna(inplace=True)

# Extract comments
comments = df['Comment'].values

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
x = vectorizer.fit_transform(comments)

y = df["Sentiment"]

# Split data correctly
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.001, random_state=40)

# logistic regression
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Logistic Regression :- Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
print(classification_report(y_test, y_pred))