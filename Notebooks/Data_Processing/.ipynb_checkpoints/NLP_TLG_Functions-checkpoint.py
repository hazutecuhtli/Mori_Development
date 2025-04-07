
import os, nltk, random
import regex as re
from string import punctuation
from collections import Counter
from gensim.models import word2vec
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
import warnings
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt


nltk.download('wordnet')
nltk.download('stopwords')
snow_stemmer = SnowballStemmer(language='spanish')
stopwords = set(stopwords.words('spanish'))
stemmer = SnowballStemmer('spanish')
lemmatizer = WordNetLemmatizer()

def tokenizer(texto):

    '''
    Function that tokenize sentences, based on the regex module

    input:

    texto -> String to tokenize

    output:

    output -> Tokens found

    '''
    
    return re.findall(r'[\w-]*\p{L}[\w-]*', texto)

    
def quita_stopword(palabras, type_='steamer'):

    '''
    Function that remove stopwords such as artilces and prepositions from the input text,
    the function also represents similar words or diffeernt word representation as a unique word,
    using a lemmatizer.

    input:

    palabras -> List of tokenized sentences or strings

    output:

    output -> List of tokenized wothds without stopwords and with unique words representations

    '''
    if type_=='steamer':
        return [stemmer.stem(palabra) for palabra in palabras if palabra.lower() not in stopwords]

    elif type_=='lemmatizer':        
        return [lemmatizer.lemmatize(palabra) for palabra in palabras if palabra.lower() not in stopwords]


def crea_corpus(textos, tipo='steamer'):

    '''
    Function that create a corpus from a list of tokenized sentences

    input:

    texto -> pandas series composed by tokenized sentences

    ouitput:

    corpus -> String composed by all the tokenized sentences

    '''

    corpus=[]
    textos=textos.values.tolist()
    corpus=[palabra.lower() for oracion in textos for palabra in oracion]
    corpus = quita_stopword(corpus, type_=tipo)
    return corpus


def nube_palabras(corpus, ax, color, stopwords=stopwords):

    '''
    Function to plot words composing a corpus ising wordcloud

    inputs:

    corpus -> Corpus
    ax -> matplotlib axis in which a figure can be plotted
    color -> Color to display the backgroud of the cloud
    stopwords -> Workds not be displayed

    '''
    
    stopwords = set(stopwords)
    wordcloud = WordCloud(
    background_color=color,
    stopwords=stopwords,
    max_words=100,
    max_font_size=45,
    scale=3,
    random_state=1)
    wordcloud=wordcloud.generate(str(corpus))
    ax.imshow(wordcloud)

def nubes_de_palabras(corpuses, colors, figsize=(10,5), stopwords=stopwords):

    '''
    Function to plot words composing a corpus ising wordcloud

    inputs:

    corpus -> Corpus
    ax -> matplotlib axis in which a figure can be plotted
    color -> Color to display the backgroud of the cloud
    stopwords -> Workds not be displayed

    '''

    stopwords = set(stopwords)
    wordcloud1 = WordCloud(background_color=colors[0],
                          stopwords=stopwords,
                          max_words=64,
                          max_font_size=45,
                          scale=3,
                          random_state=1)

    wordcloud2 = WordCloud(background_color=colors[1],
                          stopwords=stopwords,
                          max_words=128,
                          max_font_size=45,
                          scale=3,
                          random_state=1)    
    
    wordcloud1 = wordcloud1.generate(str(corpuses[0]))
    wordcloud2 = wordcloud2.generate(str(corpuses[1]))
    
    
    fig, axs = plt.subplots(2,1, figsize=figsize)  
    
    axs[0].imshow(wordcloud1)
    axs[0].set_title("Social Dataset".capitalize())
    axs[0].set_xlabel('Count')
    axs[0].set_ylabel('Words')

    axs[1].imshow(wordcloud2)
    axs[1].set_title("Technical Dataset".capitalize())
    axs[1].set_xlabel('Count')

    plt.suptitle('Most Common Words')
    fig.tight_layout()