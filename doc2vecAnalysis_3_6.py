# -*- coding: utf-8 -*-

import os
import pickle
import MeCab
import gensim
import collections
from wordcloud import WordCloud
import multiprocessing as mp
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans

def create_gensim_dictionary(data_path, mecab_path=None, no_below=2, no_above=0.1):

    if mecab_path is None:
        mecab = MeCab.Tagger("")
    else:
        mecab = MeCab.Tagger(mecab_path)

    for root, dirs, files in os.walk(data_path):
        print("# morphological analysis")
        docs = {}
        docs_title = {}
        for docname in files:
            docs[docname] = []
            with open(os.path.join(data_path, docname), "r") as f:
                lines = f.readlines()
                docs_title[docname] = lines[0]
                for text in lines:
                    res = mecab.parseToNode(text)
                    while res:
                        arr = res.feature.split(",")
                        res = res.next
                        if arr[0] != "名詞":
                            continue
                        elif len(arr[6]) == 1:
                            continue
                        else:
                            word = arr[6]
                            docs[docname].append(word)

    dictionary = gensim.corpora.Dictionary(docs.values())
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    return docs, docs_title, dictionary

def doc2vec(docs):

    print("# Start Doc2Vec")
    sentences = []
    for doc_name, word_list in docs.items():
        sentence = gensim.models.doc2vec.LabeledSentence(words=word_list, tags=[doc_name])
        sentences.append(sentence)

    model = gensim.models.Doc2Vec(seed=0, dm=1, size=300, window=5, min_count=1, sample=1e-6, iter=600)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

    model.save("Data/model/doc2vec_dmpv.model")
    
    return model

def doc2feature(model):

    print("# Start docs to feature dense")
    dense = np.zeros((len(model.docvecs), len(model.docvecs[0])), float)
    id2doc = {}
    for index in range(len(model.docvecs)):
        dense[index] = model.docvecs[index]
        id2doc[index] = model.docvecs.index_to_doctag(index)

    return dense, id2doc

def kmeans(dense, docs_title, id2doc, start, end):

    print("# Start KMeans clustering")
    cluster_result = {}
    kmeans_inertia = []
    for cluster in range(start, end):
        cluster_result[cluster] = {}
        kmeans_model = KMeans(n_clusters=cluster,
                              init="k-means++",
                              n_init=10,
                              max_iter=300,
                              tol=1e-04,
                              random_state=0)
        kmeans_model.fit(dense)
        labels = kmeans_model.labels_
        kmeans_inertia.append(kmeans_model.inertia_)

        with open("Data/clustering/doc2vec/clustering_{0}_result.txt".format(cluster), "w") as f:
            f.write("clustering {0} evaluation value : {1} \n".format(cluster, kmeans_inertia[cluster-2]))
            for id, label in enumerate(labels):
                if label not in cluster_result[cluster].keys():
                    cluster_result[cluster][label] = []
                cluster_result[cluster][label].append(id2doc[id])
                f.write("{0} is cluster {1} \n".format(id2doc[id], label))

        with open("Data/clustering/doc2vec/clustering_result_{0}_list.txt".format(cluster), "w") as f:
            f.write("clustering {0} result list\n".format(cluster))
            for cluster, docs in cluster_result[cluster].items():
                f.write("\n\n[cluster {0}] : \n".format(cluster))
                for doc in docs:
                    f.write("{0} : {1}".format(doc, docs_title[doc]))

    return cluster_result, kmeans_inertia

def subproc(p):

    ini = int(L * p / proc)
    fin = int(L * (p + 1) / proc)
    if ini == 0:
        ini = 2
    for num_topics in range(ini, fin):
        lda_model = "Data/model/multitest/model_topic{0}.lda".format(num_topics)
        cloud_name = "Data/wordcloud/multitest/wordcloud_{0}_topic".format(num_topics)
        corpus_lda_tfidf, topic_terms = lda(dictionary, corpus, corpus_tfidf, save_name=lda_model, num_topics=num_topics)
        #corpus_lda_tfidf, topic_terms = lda(dictionary, corpus, corpus_tfidf, lda_model=lda_model, num_topics=num_topics)
        docs_topic = doc2topic_id(corpus_lda_tfidf)
        _, topic_freq_word, topic_terms_20_above = word_cloud_list(dictionary, corpus_tfidf, docs_topic, topic_terms)
        create_wordcloud(topic_freq_word, topic_terms_20_above, cloud_name, "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")

def subprocess(ini, fin):

    for num_topics in range(ini, fin):
        lda_model = "Data/model/multitest/model_topic{0}.lda".format(num_topics)
        cloud_name = "Data/wordcloud/multitest/wordcloud_{0}_topic".format(num_topics)
        corpus_lda_tfidf, topic_terms = lda(dictionary, corpus, corpus_tfidf, save_name=lda_model, num_topics=num_topics)
        #corpus_lda_tfidf, topic_terms = lda(dictionary, corpus, corpus_tfidf, lda_model=lda_model, num_topics=num_topics)
        docs_topic = doc2topic_id(corpus_lda_tfidf)
        _, topic_freq_word, topic_terms_20_above = word_cloud_list(dictionary, corpus_tfidf, docs_topic, topic_terms)
        create_wordcloud(topic_freq_word, topic_terms_20_above, cloud_name, "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")

def multiprocess_1(subprocess, arg1, arg2, arg3):

    print("# Start Multi process")
    ps = [
        mp.Process(target=subprocess, args=(arg1, arg2, arg3, 2, 11)),
        mp.Process(target=subprocess, args=(arg1, arg2, arg3, 11, 21)),
        mp.Process(target=subprocess, args=(arg1, arg2, arg3, 21, 31)),
        mp.Process(target=subprocess, args=(arg1, arg2, arg3, 31, 41)),
        mp.Process(target=subprocess, args=(arg1, arg2, arg3, 41, 51)),
        mp.Process(target=subprocess, args=(arg1, arg2, arg3, 51, 61)),
        mp.Process(target=subprocess, args=(arg1, arg2, arg3, 61, 71)),
        mp.Process(target=subprocess, args=(arg1, arg2, arg3, 71, 81)),
        mp.Process(target=subprocess, args=(arg1, arg2, arg3, 81, 91)),
        mp.Process(target=subprocess, args=(arg1, arg2, arg3, 91, 101))]

    for p in ps:
        p.start()

def multiprocess_2(subprocess):

    print("# Start Multi process")
    L = 100
    proc = 10
    pool = mp.Pool(proc)
    pool.map(subprocess, range(10))



if __name__ == "__main__":

    print("# start")

    docs, docs_title, dictionary = create_gensim_dictionary("/home/tamashiro/AI/OPC/LDAPython/Data/対応方法", mecab_path=" -d /usr/lib/mecab/dic/mecab-ipadic-neologd")
    model = doc2vec(docs)

    dense, id2doc = doc2feature(model)
    cluster_result = kmeans(dense, docs_title, id2doc, 2, 101)
    #multiprocess_1(kmeans, dense, doc2id, id2doc)

    print("# END")
