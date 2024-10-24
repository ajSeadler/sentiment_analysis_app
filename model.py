import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

class SentimentModel:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def predict(self, text):
        score = self.sia.polarity_scores(text)
        return score  # Return the full score dictionary for further processing
