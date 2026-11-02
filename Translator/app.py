"""Flask web app for the Language Translation Tool."""
from flask import Flask, jsonify, render_template, request

from translator import get_languages, translate_text

app = Flask(__name__)
LANGUAGES = get_languages()  # {'english': 'en', ...}


@app.route("/")
def index():
    languages = sorted(LANGUAGES.items())
    return render_template("index.html", languages=languages)


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json(silent=True) or {}
    try:
        result = translate_text(
            data.get("text", ""),
            data.get("source", "auto"),
            data.get("target", "en"),
        )
        return jsonify(translation=result)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception:
        return jsonify(error="Translation failed. Check your internet connection and try again."), 500


if __name__ == "__main__":
    app.run(debug=True)
