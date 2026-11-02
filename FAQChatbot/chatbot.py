"""Core FAQ matching logic: NLP preprocessing + TF-IDF + cosine similarity."""
import json
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def preprocess(text: str) -> str:
    """Lowercase, remove punctuation, tokenize, remove stopwords, stem."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = text.split()
    tokens = [STEMMER.stem(t) for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


class FAQChatbot:
    def __init__(self, faq_path="faqs.json", threshold=0.25):
        with open(faq_path, encoding="utf-8") as f:
            self.faqs = json.load(f)
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer()
        corpus = [preprocess(item["question"]) for item in self.faqs]
        self.faq_matrix = self.vectorizer.fit_transform(corpus)

    def get_answer(self, user_question: str) -> str:
        cleaned = preprocess(user_question)
        if not cleaned:
            return "Please type a question."
        user_vec = self.vectorizer.transform([cleaned])
        scores = cosine_similarity(user_vec, self.faq_matrix)[0]
        best = scores.argmax()
        if scores[best] < self.threshold:
            return "Sorry, I couldn't find an answer to that. Try rephrasing your question."
        return self.faqs[best]["answer"]
