from flask import Flask, render_template, request, jsonify
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

app = Flask(__name__)

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text')

    # Calculate the overall sentiment score for the entire text
    total_score = sia.polarity_scores(text)['compound']

    # Initialize overall sentiment
    if total_score >= 0.05:
        overall_sentiment = 'Good'
    elif total_score <= -0.05:
        overall_sentiment = 'Bad'
    else:
        overall_sentiment = 'Neutral'

    # Prepare to highlight words
    words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)  # Extract words and punctuation using regex
    highlighted_words = []
    negation = False  # Track negation
    negation_words = ["not", "no", "never", "nobody", "none"]

    # Keywords and phrases that indicate negative sentiment
    negative_phrases = [
        "not clean", "lacking", "subpar", "less enjoyable", "disappointing"
    ]

    for word in words:
        # Check for negation words
        if word.lower() in negation_words:
            negation = True
            highlighted_words.append(f'<span style="color: white;">{word}</span>')  # Neutral color for negation
            continue

        # Calculate the sentiment score for the word
        word_score = sia.polarity_scores(word)['compound']
        if negation:
            word_score = -word_score  # Flip score if preceded by a negation
            negation = False  # Reset negation

        # Highlight words based on sentiment
        if word_score >= 0.05:
            highlighted_words.append(f'<span style="color: green; font-weight: bold;">{word}</span>')
        elif word_score <= -0.05:
            highlighted_words.append(f'<span style="color: red; font-weight: bold;">{word}</span>')
        else:
            highlighted_words.append(f'<span style="color: white;">{word}</span>')

    # Analyze overall context for known negative phrases
    for phrase in negative_phrases:
        if phrase in text:
            total_score -= 0.5  # Adjust the score negatively for known bad phrases

    # Adjust overall sentiment based on the revised scoring
    if total_score >= 0.05:
        overall_sentiment = 'Good'
    elif total_score <= -0.05:
        overall_sentiment = 'Bad'
    else:
        overall_sentiment = 'Neutral'

    highlighted_result = ' '.join(highlighted_words)

    return jsonify({'highlighted_text': highlighted_result, 'overall_sentiment': overall_sentiment})

if __name__ == '__main__':
    app.run(debug=True)
# if you want to host on your network then do this 

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001,debug=True)