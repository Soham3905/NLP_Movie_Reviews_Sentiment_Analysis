# Movie Mood Meter

Movie Mood Meter is a modern web app for analyzing the sentiment of movie reviews using machine learning (SVM + TF-IDF). Paste any review and get an instant prediction with confidence score.

## Features
- Clean, unique UI (not copied)
- Fast, accurate sentiment analysis
- Easy to run locally or deploy

## Project Structure

```text
my_movie_sentiment_app/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── LICENSE             # License file
├── README.md           # Project documentation
├── .gitignore          # Git ignore rules
├── ml_models/          # Trained ML models (.pkl files)
│   ├── svm_model.pkl
│   └── tfidf_vectorizer.pkl
└── templates/
    └── index.html      # Web interface
```

## Getting Started

1. **Clone this repository:**

   ```sh
   git clone https://github.com/yourusername/my_movie_sentiment_app.git
   cd my_movie_sentiment_app
   ```
2. **Create a virtual environment (recommended):**

   ```sh
   python -m venv venv
   venv\Scripts\activate
   ```
3. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```
4. **Run the app:**

   ```sh
   python app.py
   ```
5. **Open your browser:**
   Go to [http://127.0.0.1:10000](http://127.0.0.1:10000)

## Deployment

- You can deploy this app to platforms like Heroku, Render, or Azure Web Apps.
- For production, use a WSGI server (e.g., gunicorn) and set up environment variables as needed.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
