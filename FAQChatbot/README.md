# CodeAlpha FAQ Chatbot

A chatbot that answers frequently asked questions using NLP and cosine similarity.

## How it works
1. FAQs are stored in `faqs.json` (question + answer pairs).
2. Text is preprocessed with NLTK: lowercase, punctuation removal, tokenization, stopword removal, stemming.
3. Questions are converted to TF-IDF vectors (scikit-learn).
4. The user's question is matched to the most similar FAQ using **cosine similarity**.
5. If the similarity is below a threshold, the bot asks the user to rephrase.

## Technologies
Python, NLTK, scikit-learn, Flask

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_FAQChatbot.git
cd CodeAlpha_FAQChatbot
pip install -r requirements.txt
```

## Run
Command line:
```bash
python main.py
```
Web chat UI:
```bash
python app.py
```
Then open http://127.0.0.1:5000

## Project structure
```
chatbot.py        # preprocessing + similarity matching
main.py           # terminal interface
app.py            # Flask web interface
faqs.json         # FAQ data
requirements.txt
```

## Author
Your Name
