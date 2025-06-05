from flask import Flask, request, render_template, session
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from googleapiclient.discovery import build
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
import io
import base64
from collections import Counter
import urllib.parse as urlparse

# Flask App
app = Flask(__name__)

app.secret_key = 'my-sentiment-analysis'

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load ML model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# YouTube API setup
API_KEY = "AIzaSyBq9NoxEKtC_keDNlOyO37pXVKLZJCqBHc"
youtube = build("youtube", "v3", developerKey=API_KEY)

# === Utility Functions ===

def clean_comment(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

def extract_video_id(url):
    parsed = urlparse.urlparse(url)
    if "youtu.be" in url:
        return parsed.path[1:]
    return urlparse.parse_qs(parsed.query).get("v", [None])[0]

def fetch_comments(video_id):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id,
            maxResults=100, textFormat="plainText"
        )
        while request:
            response = request.execute()
            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
            if "nextPageToken" in response:
                request = youtube.commentThreads().list(
                    part="snippet", videoId=video_id,
                    maxResults=100, textFormat="plainText",
                    pageToken=response["nextPageToken"]
                )
            else:
                break
    except Exception as e:
        comments.append("Error: " + str(e))
    return comments

def fetch_video_info(video_id):
    try:
        request = youtube.videos().list(part="snippet,statistics", id=video_id)
        response = request.execute()
        if response["items"]:
            video = response["items"][0]
            snippet = video["snippet"]
            stats = video["statistics"]
            return {
                "title": snippet.get("title", "Unknown"),
                "channel": snippet.get("channelTitle", "Unknown"),
                "views": stats.get("viewCount", "N/A"),
                "thumbnail": snippet["thumbnails"]["high"]["url"]
            }
    except Exception as e:
        print("Error fetching video info:", e)
    return {}

def generate_bar_chart(sentiment_count):
    labels = ['Positive', 'Negative', 'Neutral']
    counts = [
        sentiment_count.get('Positive', 0),
        sentiment_count.get('Negative', 0),
        sentiment_count.get('Neutral', 0)
    ]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, counts, color=['#27ae60', '#e74c3c', '#f39c12'])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, int(yval), ha='center')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return chart

def generate_pie_chart(sentiment_count):
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [
        sentiment_count.get('Positive', 0),
        sentiment_count.get('Negative', 0),
        sentiment_count.get('Neutral', 0)
    ]
    colors = ['#27ae60', '#e74c3c', '#f39c12']
    plt.figure(figsize=(4, 4))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    pie = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return pie

# === Main Route ===

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    sentiment_count = {}
    total_comments = 0
    video_info = None
    bar_chart = None
    pie_chart = None
    yt_url = ""

    if request.method == "POST":
        # Get video URL from form
        yt_url = request.form.get("url", "")
        sentiment_filter = request.form.get("filter", "all")  # this is the button clicked

        # Extract video ID
        video_id = extract_video_id(yt_url)

        if video_id:
            comments = fetch_comments(video_id)
            video_info = fetch_video_info(video_id)

            cleaned = [clean_comment(c) for c in comments]
            vectors = vectorizer.transform(cleaned)
            predictions = model.predict(vectors)
            predictions = [p.capitalize() for p in predictions]

            sentiment_count = Counter(predictions)
            total_comments = len(predictions)

            bar_chart = generate_bar_chart(sentiment_count)
            pie_chart = generate_pie_chart(sentiment_count)

            results = list(zip(comments, predictions))
        else:
            results = [("Invalid YouTube URL", "N/A")]

    return render_template(
        "index.html",
        results=results,
        video_info=video_info,
        sentiment_count=dict(sentiment_count),
        total_comments=total_comments,
        bar_chart=bar_chart,
        pie_chart=pie_chart,
        yt_url=yt_url
    )


if __name__ == "__main__":
    app.run(debug=True)
