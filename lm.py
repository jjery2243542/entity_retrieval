import json
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tokenize.moses import MosesTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from scipy.sparse import csr_matrix
import string
import pickle 
import numpy as np
from collections import defaultdict
import operator
from nltk.stem.lancaster import LancasterStemmer
import argparse

def remove_punc(word):
    new_word = ''.join([c for c in word if c not in string.punctuation])
    return new_word

def preprocess(json_path='../Project1_data/DBdoc.json', query_path='../Project1_data/queries-v2.txt'):
    st = LancasterStemmer()
    stop_words = set(stopwords.words('english'))
    # processing db
    with open(json_path) as f_json:
        db = json.load(f_json)
    entitys, documents = [], []
    for entry in db:
        if entry['abstract'] != None:
            entitys.append(entry['entity'])
            words = [remove_punc(word.lower()) for word in entry['abstract'].strip().split()]
            words = [st.stem(word) for word in words if not word.isdigit()]
            documents.append(' '.join(words))
    # processing query

    query_ids, queries = [], []
    with open(query_path) as f_q:
        for line in f_q:
            query_id, raw_query = line.strip().split('\t')
            query_ids.append(query_id)
            words = [remove_punc(word.lower()) for word in raw_query.strip().split()]
            words = [st.stem(word) for word in words if not word.isdigit()]
            queries.append(' '.join(words))
    return entitys, documents, query_ids, queries

def get_terms(documents, n_terms=30000):
    count_dict = defaultdict(lambda : 0)
    for document in documents:
        for word in document.split():
            count_dict[word] += 1
    sort_terms = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)
    stop_words = set(stopwords.words('english'))
    terms = []
    for term, _ in sort_terms:
        if len(terms) >= n_terms:
            break
        if term not in stop_words and not term.isdigit():
            terms.append(term)
    return terms

def get_lm(documents, word2idx):
    alpha = 0.2
    lms = []
    for document in documents:
        lm = np.zeros(len(word2idx), dtype=np.float32)
        for word in document.split():
            if word in word2idx:
                idx = word2idx[word]
                lm[idx] += 1
        lms.append(lm)
    lms = np.array(lms)
    background_lm = np.sum(lms, axis=0)
    total = np.sum(background_lm)
    background_lm /= total
    total = np.expand_dims(np.sum(lms, axis=1), axis=1)
    lms = lms / (total + 1e-12)
    lms = alpha * lms + (1 - alpha) * background_lm
    lms = np.log(lms)
    return lms

def topk_lm_score(lms, word2idx, query, k=100):
    # dealing with single query
    word_idxs = [word2idx[word] for word in query.split() if word in word2idx]
    scores = np.sum(lms[:,word_idxs], axis=1)
    rank = np.argsort(scores)
    topk = np.fliplr([rank[-k:]])[0]
    sorted_scores = scores[topk]
    return topk, sorted_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-db_path', default='../Project1_data/DBdoc.json')
    parser.add_argument('-query_path', default='../Project1_data/queries-v2.txt')
    parser.add_argument('-output_path', default='./predict-rel.txt')
    args = parser.parse_args()
    # preprocessing
    entitys, documents, query_ids, queries = preprocess(args.db_path, args.query_path)
    # choose most frequent 30000 terms
    terms = get_terms(documents)
    # mapping word to index
    word2idx = {term:idx for idx, term in enumerate(terms)}
    # calculate uni-gram language model
    lms = get_lm(documents, word2idx)
    with open(args.output_path, 'w') as f:
        for query_id, query in zip(query_ids, queries):
            # find top-k documents
            topk_document_ids, scores = topk_lm_score(lms, word2idx, query)
            for rank_m1, (document_id, score) in enumerate(zip(topk_document_ids, scores)):
                f.write('{}\tQ0\t<dbpedia:{}>\t{}\t{}\tSTANDARD\n'.format(
                    query_id, entitys[document_id], rank_m1+1, np.exp(score)
                ))
