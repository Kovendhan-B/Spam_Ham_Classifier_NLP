import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

stopwords_set = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    cleaned_text = ''.join([char if char.isalpha() or char.isspace() else ' ' for char in text])
    tokens = cleaned_text.split() 
    tokens = [word for word in tokens if word not in stopwords_set]  
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)


with open(r"D:\ML\Projects\Spam_Ham_Classifier_NLP\random_forest_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

with open(r"D:\ML\Projects\Spam_Ham_Classifier_NLP\vectorizer.pkl", 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_and_vectorize(text):
    preprocessed_text = preprocess_text(text)
    return vectorizer.transform([preprocessed_text])

def predict(text):
    vectorized_text = preprocess_and_vectorize(text)
    prediction = model.predict(vectorized_text)
    return "Spam" if prediction[0] == 1 else "Ham"

email = input("Enter Email text: ")
print(predict(email))
