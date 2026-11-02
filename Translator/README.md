# CodeAlpha Language Translation Tool

A web app that translates text between more than 100 languages using Google Translate.

## Features
- Enter text and choose the source language (or auto-detect) and the target language
- Translation is requested from Google Translate and displayed clearly
- Swap languages button
- Copy button
- Text-to-speech (Listen) button
- Input validation and error handling
- Supports right-to-left languages such as Arabic

## Technologies
Python, Flask, deep-translator (Google Translate), HTML/CSS/JavaScript

## Installation
```bash
git clone https://github.com/shimaa-mansour/codealpha_tasks.git
cd codealpha_tasks/Translator
pip install -r requirements.txt
```

## Run
```bash
python app.py
```
Then open http://127.0.0.1:5000 (an internet connection is required).

## Project structure
```
Translator/
├── app.py             # Flask server
├── translator.py      # translation logic
├── templates/
│   └── index.html     # user interface
└── requirements.txt
```

## Author
Shimaa Mansour
