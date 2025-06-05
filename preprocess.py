import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv("US comments.csv", encoding='latin1', low_memory=False)
comments = df['comment_text']

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_comment(comment):
    if not isinstance(comment, str):
        return " "

    # Lowercase
    comment = comment.lower()

    # Remove URLs
    comment = re.sub(r'http\S+|www\S+|https\S+', '', comment)

    # Remove punctuation and numbers
    comment = re.sub(r'[^a-zA-Z\s]', '', comment)

    # Remove extra spaces
    comment = re.sub(r'\s+', ' ', comment).strip()

    # Remove stopwords and lemmatize
    words = comment.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)

df_generate = df['comment_text'].dropna().astype(str).apply(clean_comment)


df_generate.to_csv("new comment.csv")