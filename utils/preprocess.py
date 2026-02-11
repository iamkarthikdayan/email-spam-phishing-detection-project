import re
from bs4 import BeautifulSoup
import nltk
from  nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words=set(stopwords.words('english'))


def clean_email(text):
    soup=BeautifulSoup(text, "html.parser")
    text=soup.get_text()


    text=text.lower()

    text=re.sub(r'http\S+',  '', text)

    text=re.sub(r'[^a-zA-Z0-9\s]', '', text)

    words=word_tokenize(text)

    filtered_words= [word for word in words if word not in stop_words]

    return ' '.join(filtered_words)

   