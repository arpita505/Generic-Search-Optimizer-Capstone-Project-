from __future__ import division
import re
import math
import string
import pandas as pd
import numpy as np
from collections import Counter

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

data_file = pd.read_excel('data_f.xlsx')
from django.utils.encoding import smart_str
book_names = [smart_str(n).lower() for n in data_file['title'] if n!=0]

from nltk.tokenize import RegexpTokenizer
import nltk
tokenizer = RegexpTokenizer(r'\w+')
tokens = [tokenizer.tokenize(i) for i in book_names]

import xlrd
workbook = xlrd.open_workbook('data_f.xlsx', on_demand = True)
sheet=workbook.sheet_by_name('Sheet1')

dict=open('dicttotalwithoutnums.txt').read()

def onlytext(text):
    return re.findall('[a-z]+', text.lower())

words=onlytext(dict)
count =[len(item) for item in tokens]
n=sum(count)/len(count)
Counter(dict)
COUNTS=Counter(words)

def correct(word):
    # Preference order edit distance 0->1->2 else def.
    candidates = (known(edits0(word)) or 
                  known(edits1(word)) or 
                  known(edits2(word)) or 
                  [word])
    return max(candidates, key=COUNTS.get)

def known(words):
    return {w for w in words if w in COUNTS}
def edits0(word):
    return {word}
def edits2(word):
    return { ed2 for ed1 in edits1(word)for ed2 in edits1(ed1)
    }

def edits1(word):
    pairs      = splits(word)
    deletes    = [a+b[1:]           for (a, b) in pairs if b]
    transposes = [a+b[1]+b[0]+b[2:] for (a, b) in pairs if len(b) > 1]
    replaces   = [a+c+b[1:]         for (a, b) in pairs for c in alphabet if b]
    inserts    = [a+c+b             for (a, b) in pairs for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def splits(word):
    #Return a list of all possible (first, rest) pairs that comprise word
    return [(word[:i], word[i:]) 
            for i in range(len(word)+1)]

alphabet = 'abcdefghijklmnopqrstuvwxyz'


def correct_text(text):
    return re.sub('[a-zA-Z]+', correct_match, text)
#preserve case and correct
def correct_match(match):
    word = match.group()
    return case_of(word)(correct(word.lower()))
#case-function appropriate
def case_of(text):
    return (str.upper if text.isupper() else
            str.lower if text.islower() else
            str.title if text.istitle() else
            str) 

whole_words = [' '.join(tokens[i]) for i in range(len(tokens))]
whole_set_of_words = ' '.join(whole_words)

#Make a probability distribution,using Counter
def pdist(counter):
    N = sum(counter.values())
    return lambda x: counter[x]/N

P = pdist(COUNTS)


def Pwords(words):
    return product(P(w) for w in words)
def product(nums):
    result = 1
    for x in nums:
        result *= x
    return result

#Memoize function f, whose args must all be hashable.
def memo(f):
    cache = {}
    def fmemo(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]
    fmemo.cache = cache
    return fmemo

def splits(text, start=0, L=14):
    return [(text[:i], text[i:]) 
            for i in range(start, min(len(text), L)+1)]

def segment(text):
    st = ""
    found = False
    start = 0
    end = len(text)
    if (' ' in text) == True:
        for i,c in enumerate(text):
            if c==" ":
                st = st + text[start:i]
                start = i+1
                found = True
    if found:
        st = st+ text[start:end]
        text = st
    if not text: 
        return []
    else:
        candidates = ([first] + segment(rest) 
                      for (first, rest) in splits(text, 1))
        return max(candidates, key=Pwords)


def load_counts(filename, sep='\t'):
    C = Counter()
    for line in open(filename):
        key, count = line.split(sep)
        C[key] = int(count)
    return C

COUNTS1 = load_counts('count_1f.txt')
COUNTS2 = load_counts('count_2f.txt')

P1w = pdist(COUNTS1)
P2w = pdist(COUNTS2)


def Pwords2(words, prev='<S>'):
    return product(cPword(w, (prev if (i == 0) else words[i-1]) )
                   for (i, w) in enumerate(words))
# Change Pwords to use P1w (the bigger dictionary) instead of Pword
def Pwords(words):
    return product(P1w(w) for w in words)
#Conditional probability of word, given previous word
def cPword(word, prev):    
    bigram = prev + ' ' + word
    if P2w(bigram) > 0 and P1w(prev) > 0:
        return P2w(bigram) / P1w(prev)
    else: # Average the back-off value and zero.
        return P1w(word) / 2

@memo 
def segment2(text, prev='<S>'): 
    "Return best segmentation of text; use bigram data." 
    st = ""
    found = False
    start = 0
    end = len(text)
    if (' ' in text) == True:
        for i,c in enumerate(text):
            if c==" ":
                st = st + text[start:i]
                start = i+1
                found = True
    if found:
        st = st+ text[start:end]
        text = st

    if not text: 
        return []
    else:
        candidates = ([first] + segment2(rest, first) 
                      for (first, rest) in splits(text, 1))
        return max(candidates, key=lambda words: Pwords2(words, prev))

import nltk
from nltk.corpus import stopwords
stp_words = set(stopwords.words('english'))
for book_name in tokens:
    for word in book_name[:]:
        if word in stp_words or word.isdigit():
            book_name.remove(word)


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

num_features = 600   # Word vector dimensionality
min_word_count = 1 # Minimum word count
num_workers = 6    # Number of threads to run in parallel
context = 200         # Context window size                                                                                    
downsampling = 1e-3


from gensim.models import word2vec
model = word2vec.Word2Vec(tokens, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

def myset(lis):
    final_lis = []
    for item in lis:
        if item not in final_lis:
            final_lis.append(item)
    return final_lis

def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if type(item) == type([]):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis
import re
t=[]
def sim(it):
    lst2=[]
    try:
        data_from_model = model.most_similar(it)
        #lst2.append([str(item[0]) for item in t])
        str_of_items = str(it)
        for item in data_from_model:
            str_of_items = str_of_items + ' ' + smart_str(item[0])
        item_found = False
        while item_found == False:
            word_item = model.doesnt_match(str_of_items.split())
            if not(word_item == str(it)):
                str_of_items = str_of_items[0: str_of_items.index(word_item)-1] + str_of_items[str_of_items.index(word_item) + len(word_item):]
            else:
                item_found = True
        #lst2.append(it)
        #lst2=(flatten(lst2))
        print(str_of_items)
        matching=[]
        for item in str_of_items.split():
            for s in book_names:
                if " "+item+" " in s:
                    matching.append(s)
    except:
        matching=[]
    return list(myset(matching))

labels=pd.read_excel('data_f.xlsx')
dc = labels.ix[:,['Subjects']]
def ddc(it):
    book=[]
    i=0
    for index, row in labels.iterrows():
        x=smart_str(row['Subjects']).strip(',')
        g=re.search(it,x, flags=0)
        if g :
            try:
                book.insert(i,row['title'].lower())
                i=i+1
            except:
                book.insert(i,row['title'])
                i=i+1
    return list(myset(book))

def inter(l1,l2):
    list_new= [item for item in l1 if item in l2]
    return list_new

def tt(it):
    common=[]
    book_ddc=ddc(it)
    kaggel=sim(it)
    common=inter(book_ddc,kaggel)
    if not common:
        print("no Common words present in both the models...\n")
        common= list(kaggel + book_ddc)
    print(len(kaggel))
    print(len(book_ddc))
    print(len(common))
    return common

def extract_books_direct(inp_name):
    data_file = pd.read_excel('data_f.xlsx')
    book_names = [smart_str(n) for n in data_file['title'] if n!=0]
    query_matched_data = []
    for b_name in book_names:
        if inp_name in b_name:
            query_matched_data.append(b_name)
    return query_matched_data

def list_of_books(str1):
    str1 = str1.lower()
    r=re.findall(r'\w+', str1)
    final=[]
    final = list(myset(extract_books_direct(str1)))
    tags = nltk.pos_tag(r)
    for word, tag in tags:
        print(word)
        print(tag)
        if tag.startswith('N') or tag.startswith('V'):
            f_t=tt(word)
            if f_t:
                final.append(f_t)
    final=flatten(final)
    print(len(final))
    return list(myset(final))

from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')
@app.route("/",methods=['POST'])
def Submit():
    _name = request.form['inputName']
    str1 = smart_str(_name)
    print(str1)
    str3 = correct_text(str1)
    str4 = segment2(str3)
    print(str4)
    print(str3)
    data1=list_of_books(smart_str(str4))
    
    if _name :
        return render_template('second.html',your_list=data1,text=str4,text2=len(data1))
if __name__ == "__main__":
    app.run()



