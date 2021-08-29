# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import nltk
paragraph = """Data science is an interdisciplinary field that uses scientific metho
ds, processes, algorithms and systems to extract knowledge and insights from structu
red and unstructured data,[1][2] and apply knowledge and actionable insights 
from data across a broad range of application domains. Data science is related 
to data mining, machine learning and big data.
Data science is a "concept to unify statistics, data analysis, informatics, 
and their related methods" in order to "understand and analyze actual phenomena"
 with data.[3] It uses techniques and theories drawn from many fields within 
 the context of mathematics, statistics, computer science, information science, 
 and domain knowledge. However, data science is different from computer science 
 and information science. Turing Award winner Jim Gray imagined data science as a 
 "fourth paradigm" of science (empirical, theoretical, computational, and no
data-driven) and asserted that "everything about science is changing because 
of the impact of information technology" and the data deluge.[4][5]Data science
 is an interdisciplinary field focused on extracting knowledge from data sets, 
 which are typically large (see big data), and applying the knowledge and 
 actionable insights from data to solve problems in a wide range of application
 domains.[6] The field encompasses preparing data for analysis, formulating 
 data science problems, analyzing data, developing data-driven solutions, and 
 presenting findings to inform high-level decisions in a broad range of 
 application domains. As such, it incorporates skills from computer science, 
 statistics, information science, mathematics, information visualization, 
 data integration, graphic design, complex systems, communication and 
 business.[7][8] Statistician Nathan Yau, drawing on Ben Fry, also links 
 data science to human-computer interaction: users should be able to 
 intuitively control and explore data.[9][10] In 2015, the American Statistical
 Association identified database management, statistics and machine learning, 
 and distributed and parallel systems as the three emerging foundational 
 professional communities.[11]"""
 
 
 # Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Bag of Words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()

# Lemmatization

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)      

# Stemming
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)   
    
    
# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()


# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

# Preparing the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
    
# Word2Vec
model = Word2Vec(sentences, min_count=1)


words = model.wv.vocab

# Finding Word Vectors
vector = model.wv['data Science']

# Most similar words
similar = model.wv.most_similar('Python')
