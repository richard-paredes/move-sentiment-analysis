import click
import nltk
import pandas as pd
import pickle as pkl

from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from dataclasses import make_dataclass


DataPoint = make_dataclass("ReviewSentiment", [("review", str), ("sentiment", str)])

def download_data():
    # Run once to install data for analyzing the movie_reviews corpus.
    
    nltk.download("movie_reviews")
    nltk.download("stopwords")
    nltk.download("punkt_tab")
    nltk.download("wordnet")

def prepare_dataframe():
    reviews = [DataPoint(" ".join(movie_reviews.words(fileid)), sentiment) 
               for sentiment in movie_reviews.categories()
               for fileid in movie_reviews.fileids(sentiment)]
    
    df = pd.DataFrame(reviews)
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

def save_file(obj, file_name: str):
    with open(file_name, 'wb') as file:
        pkl.dump(obj, file)

def vectorize(df: pd.DataFrame):
    # need to learn what goes on here...
    vectorizer = TfidfVectorizer(
        max_features=2000, # Consider the top 2000 features (words/n-grams of words)
        ngram_range=(1,2), # Considers single words and pairs of words as features,
        # stop_words="english" # Automatically remove stop-words; done as part of preprocessing
    )
    X = vectorizer.fit_transform(df['review'])
    save_file(vectorizer, "vectorizer.pkl")
    return X

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2)

def train(X_train, y_train):
    model = MultinomialNB(
        fit_prior=False # Ensure equal class probabilities for documents, this way new
                        # reviews won't get a bias fitted from the % of sentiment 
                        # based solely on training data. Instead, it will come
                        # from the features of the document.
    )
    model.fit(X=X_train, y=y_train)
    return model

def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')
    conf_matrix = confusion_matrix(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion matrix: {conf_matrix}")

def create_model(output_filename: str):
    download_data()
    df = prepare_dataframe()
    df['review'] = df['review'].apply(preprocess)
    X, y = (vectorize(df), df['sentiment'])
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train(X_train, y_train)
    save_file(model, output_filename)
    evaluate(model, X_test, y_test)

def load_pretrained(filename: str):
    print(f"Loading pretrained model from {filename}")
    with open(filename, 'rb') as model_file:
        model = pkl.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pkl.load(vectorizer_file)
    return model, vectorizer

@click.command()
@click.option("--pretrained", default=True, help='Use a pretrained semantic classifer.')
def main(pretrained: bool):
    """Runs semantic analysis on a model"""
    click.echo(f'Using a pretrained model? {pretrained}')
    filename = "multinomialNB.pkl"
    if not pretrained: create_model(filename)
    model, vectorizer = load_pretrained(filename)
    review = click.prompt("Enter a review to analyze: ")
    cleaned_review = preprocess(review)
    vector = vectorizer.transform([cleaned_review])
    predicted_sentiments = model.predict(vector)
    print(f'Model predicted: {predicted_sentiments[0]}')

if __name__ == "__main__":
    main()
    
    