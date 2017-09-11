# -*- coding: utf-8 -*-

import os
import pickle
import MeCab
import gensim
import collections
from wordcloud import WordCloud

def create_gensim_dictionary(data_path, mecab_path=None, no_below=2, no_above=0.1):

    if mecab_path is None:
        mecab = MeCab.Tagger("")
    else:
        mecab = MeCab.Tagger(mecab_path)

    for root, dirs, files in os.walk(data_path):
        print("# morphological analysis")
        docs = collections.OrderedDict()
        for docname in files:
            docs[docname] = []
            with open(os.path.join(data_path, docname), "r") as f:
                lines = f.readlines()
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

    return docs, dictionary

def create_gensim_corpus(docs, dictionary):

    print("# create corpus")
    corpus = collections.OrderedDict()
    for file_name, word_list in docs.items():
        corpus[file_name] = dictionary.doc2bow(word_list)

    tfidf = gensim.models.TfidfModel(corpus.values())
    tfidf_list = tfidf[corpus.values()]

    corpus_tfidf = collections.OrderedDict()
    tfidf_list_ = list(tfidf_list)
    for id, docname in enumerate(corpus.keys()):
        corpus_tfidf[docname] = tfidf_list_[id]

    return corpus, corpus_tfidf

def create_gensim_topic_corpus(docs, dictionary, topic_docs):

    print("# create topic corpus")
    corpus_topic = collections.OrderedDict()
    for topic, files in topic_docs.items():
        if topic not in corpus_topic.keys():
            corpus_topic[topic] = collections.OrderedDict()
        for file_name in files:
            corpus_topic[topic][file_name] = dictionary.doc2bow(docs[file_name])

    topic_tfidf = collections.OrderedDict()
    for topic in topic_docs:
        tfidf = gensim.models.TfidfModel(corpus_topic[topic].values())
        topic_tfidf[topic] = tfidf[corpus_topic[topic].values()]

    corpus_topic_tfidf = collections.OrderedDict()
    for topic, corpus in corpus_topic.items():
        tfidf_list = list(topic_tfidf[topic])
        corpus_topic_tfidf[topic] = collections.OrderedDict()
        for id, file_name in enumerate(corpus.keys()):
            corpus_topic_tfidf[topic][file_name] = tfidf_list[id]

    return corpus_topic, corpus_topic_tfidf


def lda(dictionary, corpus, corpus_tfidf, lda_model=None, save_name="test.model", num_topics=2):

    print("# LDA")
    corpus_tfidf_list = corpus_tfidf.values()
    if lda_model is None:
        lda_model = gensim.models.LdaModel(corpus.values(), id2word=dictionary, num_topics=num_topics)
        lda_tfidf = lda_model[corpus_tfidf_list]
        lda_model.save(save_name)
    else:
        lda_model = gensim.models.LdaModel.load(lda_model)
        lda_tfidf = lda_model[corpus_tfidf_list]

    corpus_lda_tfidf = collections.OrderedDict()
    lda_tfidf_ = list(lda_tfidf)
    for id, docname in enumerate(corpus_tfidf.keys()):
        corpus_lda_tfidf[docname] = lda_tfidf_[id]

    return corpus_lda_tfidf

def doc2topic_id(corpus_lda_tfidf):

    print("# doc2topic_id")
    docs_topic = collections.OrderedDict()
    for doc_name, doc in corpus_lda_tfidf.items():
        docs_topic[doc_name] = max(doc, key=lambda x:x[1])[0]

    print(collections.Counter(docs_topic.values()))

    topic_docs = collections.OrderedDict()
    for docname, topic in docs_topic.items():
        if topic == 0:
            if topic not in topic_docs.keys():
                topic_docs[topic] = []
            topic_docs[topic].append(docname)
        else:
            if topic not in topic_docs.keys():
                topic_docs[topic] = []
            topic_docs[topic].append(docname)

    return docs_topic, topic_docs

def word_cloud_list(dictionary, corpus_tfidf, docs_topic):

    print("# create word cloud list")
    topic_freq_word_id = collections.OrderedDict()
    topic_freq_word = collections.OrderedDict()
    
    for docname, topic in docs_topic.items():
        for word_id, tfidf in corpus_tfidf[docname]:
            if topic not in topic_freq_word_id.keys():
                topic_freq_word_id[topic] = collections.OrderedDict()
                topic_freq_word[topic] = collections.OrderedDict()
            if word_id not in topic_freq_word_id[topic]:
                topic_freq_word_id[topic][word_id] = tfidf
                topic_freq_word[topic][dictionary[word_id]] = tfidf
            else:
                sum_tfidf = topic_freq_word_id[topic][word_id] + tfidf
                topic_freq_word_id[topic][word_id] = sum_tfidf
                topic_freq_word[topic][dictionary[word_id]] = sum_tfidf

    return topic_freq_word_id, topic_freq_word

def sort_frequency_word(topic_freq_word):

    print("# sort frequency word")
    topic_sorted = collections.OrderedDict()

    for topic, topic_words in topic_freq_word.items():
        topic_sorted[topic] = sorted(topic_words.items(), key=lambda x:x[1], reverse=True)

    return topic_sorted

def create_wordcloud(topic_freq_word, file_name, font_path, background_color="white", width=1024, height=674):

    print("# create word cloud")
    wordcloud_model = WordCloud(background_color=background_color, font_path=font_path, width=width, height=height)
    for topic, frequencies in topic_freq_word.items():
       wordcloud_obj = wordcloud_model.generate_from_frequencies(frequencies)
       save_name = "{0}{1}.png".format(file_name, topic)
       wordcloud_obj.to_file(save_name)

    print("# END")

if __name__ == "__main__":

    print("# start")
    docs, dictionary = create_gensim_dictionary("/home/tamashiro/AI/OPC/LDAPython/Data/対応方法", mecab_path=" -d /usr/lib/mecab/dic/mecab-ipadic-neologd")
    corpus, corpus_tfidf = create_gensim_corpus(docs, dictionary)
    #corpus_lda_tfidf = lda(dictionary, corpus, corpus_tfidf, save_name="Data/model/model_Noun.lda")
    corpus_lda_tfidf = lda(dictionary, corpus, corpus_tfidf, lda_model="Data/model/model_Noun.lda")
    docs_topic, topic_docs = doc2topic_id(corpus_lda_tfidf)
    corpus_topic, corpus_topic_tfidf = create_gensim_topic_corpus(docs, dictionary, topic_docs)
    #_, topic_freq_word = word_cloud_list(dictionary, corpus_tfidf, docs_topic)
    #create_wordcloud(topic_freq_word, "wordcloud_Noun_b", "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")
