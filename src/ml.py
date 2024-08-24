import nltk
import pandas as pd
from nltk.corpus import movie_reviews, stopwords
from dataclasses import make_dataclass

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


DataPoint = make_dataclass("ReviewSentiment", [("review", str), ("sentiment", str)])

def download_data():
    # Run once to install data for analyzing the movie_reviews corpus.
    
    nltk.download("movie_reviews")
    nltk.download("stopwords")
    nltk.download("punkt_tab")
    nltk.download("wordnet")

    print("Movie reviews have been downloaded")

def prepare_dataframe():
    reviews = [DataPoint(" ".join(movie_reviews.words(fileid)), sentiment) 
               for sentiment in movie_reviews.categories()
               for fileid in movie_reviews.fileids(sentiment)]
    
    df = pd.DataFrame(reviews)
    print(df)
    return df

def preprocess(review: str):
    # Tokenize the text
    tokens = word_tokenize(review.lower())
    # Remove any stop/filler words that don't add meaning to the sentence
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens, i.e. cutting any tenses and grabbing just the base word
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return " ".join(lemmatized_tokens)

def vectorize(review: str):
    # need to learn what goes on here...
    


if __name__ == "__main__":
    download_data()
    df = prepare_dataframe()
    df['review'] = df['review'].apply(preprocess)
    print(df)