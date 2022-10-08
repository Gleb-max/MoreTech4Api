import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def my_lemmatizer(sent, lemmatizer=lemmatizer):
    tokenized_sent = sent.split()
    pos_tagged = [(word, get_wordnet_pos(tag))
                 for word, tag in pos_tag(tokenized_sent)]
    return ' '.join([lemmatizer.lemmatize(word, tag)
                    for word, tag in pos_tagged])

nltk.download('punkt')

hard_string = ''
hard_string.split()
expr = r'[^(\w.\w)\w\s]'
parser=re.compile(expr)
tmp_string = parser.sub(r'', hard_string)
print(tmp_string.split())

tmp_string = re.split(r'[!.?]', hard_string)
print(tmp_string)

stemmer = SnowballStemmer(language='english')
sent = 'George admitted the talks happened'

stemmer = SnowballStemmer(language='english')
sent = 'write wrote written'

from nltk import wordnet, pos_tag
def get_wordnet_pos(treebank_tag):
    my_switch = {
        'J': wordnet.wordnet.ADJ,
        'V': wordnet.wordnet.VERB,
        'N': wordnet.wordnet.NOUN,
        'R': wordnet.wordnet.ADV,
    }
    for key, item in my_switch.items():
        if treebank_tag.startswith(key):
            return item
    return wordnet.wordnet.NOUN



