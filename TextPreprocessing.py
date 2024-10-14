import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

def tokenize_text(text):
    return word_tokenize(text)

def lowercase_text(text):
    return text.lower()

def remove_stopwords(tokens):
    # only tokenised text can be passed in this else it will give a very scrambled answer
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word.lower() not in stop_words]

def lemmatize_tokens(tokens):
    # pass removed stopword text in this 
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def stemming_tokens(tokens):
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(token) for token in tokens]

def remove_special_characters(text):
    return re.sub(r'[^A-Za-z0-9$#\s]', '', text)

def preprocess_text(text):
    text = remove_special_characters(text)
    text = lowercase_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    # tokens = lemmatize_tokens(tokens)
    tokens = stemming_tokens(tokens)
    return tokens


sample_text = "The quick brown fox, jumping over the lazy dog for $2000!"
print(preprocess_text(sample_text))
