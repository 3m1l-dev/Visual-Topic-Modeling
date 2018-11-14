import re
import nltk
import time

# Download nltk if you haven't :
# nltk.download()

from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *

"""
BASIC PRE PROCESSING 
"""

def initial_clean(text):
	"""
	Cleans text of websites, emails, punctuation and returns tokens.
	"""
	replace_by_space_re = re.compile(r'[/(){}\[\]\|@,;]')
	bad_symbols_re = re.compile('[^0-9a-z #+_]')
	stop_words = set(stopwords.words('english'))

	text = text.lower()# lowercase text
	text = replace_by_space_re.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
	text = bad_symbols_re.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text

	text = nltk.word_tokenize(text)
	return[word for word in text if word not in stop_words]

def stem_words(text):
	"""
	Function to stem words, removing 1 letter words, and including try except
	for words that can potentially break the stemmer. 
	"""
	stemmer = PorterStemmer()
	try:
		text = [stemmer.stem(word) for word in text]
		text = [word for word in text if len(word) > 1]
	except IndexError:
		pass
	return text    

def preprocess(text):
	return stem_words(initial_clean(text))

"""
VOCABULARY PREPROCESSING, INCLUDING BAG OF WORDS AND TF-IDF 
"""

def select_top_k(data, k):
	all_words = [word for item in list(data['processed']) for word in item]
	fdist = FreqDist(all_words)
	top_k_words = fdist.most_common(k)
	print("Least common words of top " + str(k))
	print(top_k_words[-10:])


def keep_top_k_words(data, k):
	all_words = [word for item in list(data['processed']) for word in item]
	fdist = FreqDist(all_words)
	top_k_words = fdist.most_common(k)	
	top_k_words,_ = zip(*fdist.most_common(k))
	top_k_words = set(top_k_words)

	def top(text):
		return [word for word in text if word in top_k_words]

	return(data['processed'].apply(top))
