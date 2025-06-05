# 🎯 YouTube Comment Sentiment Analyzer

This Flask-based web app allows users to input a YouTube video URL and analyzes the top comments using a trained machine learning model to classify sentiments as **Positive**, **Negative**, or **Neutral**. It also displays sentiment distribution using bar and pie charts.

---

## 🚀 Features

- 🔍 Extracts comments from any public YouTube video using YouTube Data API
- 🧹 Cleans and preprocesses comments using NLTK
- 🤖 Classifies sentiments using a trained ML model
- 📊 Displays visual analytics (Bar & Pie Charts) using Matplotlib
- 🎥 Shows video details like thumbnail, title, channel, and view count
- 📋 Tabulated comment-wise sentiment prediction

---

## 🛠️ Tech Stack

- **Backend**: Flask
- **ML Model**: Scikit-learn (with TF-IDF vectorizer)
- **NLP**: NLTK
- **Visualization**: Matplotlib
- **API**: YouTube Data API v3
- **Deployment**: [Render](https://render.com)

---

## 🧑‍💻 How to Run Locally

### ✅ Prerequisites

- Python 3.8+
- YouTube Data API Key (place it in the code)

### 🔧 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/youtube-sentiment-analyzer.git
   cd youtube-sentiment-analyzer
