from flask import Flask, jsonify, render_template, request
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def ensure_nltk_data():
    resources = ["punkt", "wordnet", "punkt_tab"]
    for resource in resources:
        try:
            if resource == "wordnet":
                nltk.data.find("corpora/wordnet")
            elif resource == "punkt":
                nltk.data.find("tokenizers/punkt")
            elif resource == "punkt_tab":
                nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download(resource, quiet=True)


ensure_nltk_data()

app = Flask(__name__)

knowledge_base = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hi there! What can I do for you?",
    "what is your name": "I am an AI chatbot built using Python, NLTK, and Flask.",
    "how does nlp work": "NLP (Natural Language Processing) helps computers understand, interpret, and manipulate human language.",
    "bye": "Goodbye! Have a great day ahead!",
}

questions = list(knowledge_base.keys())
responses = list(knowledge_base.values())
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    tokens = nltk.word_tokenize(str(text).lower())
    return " ".join(lemmatizer.lemmatize(token) for token in tokens if token.isalnum())


def get_bot_response(user_input):
    cleaned_user_input = clean_text(user_input)
    if not cleaned_user_input.strip():
        return "Please type something so I can help!"

    corpus = questions + [cleaned_user_input]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    best_match_idx = similarity_scores.argmax()
    highest_score = similarity_scores[0, best_match_idx]
    threshold = 0.2

    if highest_score < threshold:
        return "I'm sorry, I didn't quite understand that. Could you rephrase?"

    return responses[best_match_idx]


@app.route("/")
def home():
    return render_template("index1.html")


@app.route("/get", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = str(data.get("msg", "") or "").strip()
    bot_response = get_bot_response(user_message)
    return jsonify({"response": bot_response})


if __name__ == "__main__":
    app.run(debug=True)