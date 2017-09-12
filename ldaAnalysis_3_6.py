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
        docs = {}
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
    corpus = {}
    for file_name, word_list in docs.items():
        corpus[file_name] = dictionary.doc2bow(word_list)

    tfidf = gensim.models.TfidfModel(corpus.values())
    tfidf_list = tfidf[corpus.values()]

    corpus_tfidf = {}
    tfidf_list_ = list(tfidf_list)
    for id, docname in enumerate(corpus.keys()):
        corpus_tfidf[docname] = tfidf_list_[id]

    return corpus, corpus_tfidf

def lda(dictionary, corpus, corpus_tfidf, lda_model=None, save_name="test.model", num_topics=2):

    print("# Start LDA")
    lda_tfidf = []
    if lda_model is None:
        lda_model = gensim.models.LdaModel(corpus_tfidf.values(), id2word=dictionary, num_topics=num_topics, random_state=0)
        lda_model.save(save_name)
        for doc_tfidf in corpus_tfidf.values():
            lda_tfidf.append(lda_model[doc_tfidf])
    else:
        lda_model = gensim.models.LdaModel.load(lda_model)
        for doc_tfidf in corpus_tfidf.values():
            lda_tfidf.append(lda_model[doc_tfidf])

    corpus_lda_tfidf = {}
    for id, docname in enumerate(corpus_tfidf.keys()):
        corpus_lda_tfidf[docname] = lda_tfidf[id]
    
    topic_terms = []
    for id in range(num_topics):
        topic_terms.append(lda_model.get_topic_terms(id, topn=20))

    return corpus_lda_tfidf, topic_terms

def doc2topic_id(corpus_lda_tfidf):

    print("# doc2topic_id")
    docs_topic = {}
    for doc_name, doc in corpus_lda_tfidf.items():
        docs_topic[doc_name] = max(doc, key=lambda x:x[1])[0]

    print(collections.Counter(docs_topic.values()))

    return docs_topic

def word_cloud_list(dictionary, corpus_tfidf, docs_topic, topic_terms):

    print("# create word cloud list")
    topic_freq_word_id = {}
    topic_freq_word = {}
    topic_terms_20_above = {}

    for topic, terms in enumerate(topic_terms):
        topic_terms_20_above[topic] = {}
        for word_id, degree in terms:
            topic_terms_20_above[topic][dictionary[word_id]] = degree
    
    for docname, topic in docs_topic.items():
        for word_id, tfidf in corpus_tfidf[docname]:
            if topic not in topic_freq_word_id.keys():
                topic_freq_word_id[topic] = {}
                topic_freq_word[topic] = {}
            if word_id not in topic_freq_word_id[topic]:
                topic_freq_word_id[topic][word_id] = tfidf
                topic_freq_word[topic][dictionary[word_id]] = tfidf
            else:
                sum_tfidf = topic_freq_word_id[topic][word_id] + tfidf
                topic_freq_word_id[topic][word_id] = sum_tfidf
                topic_freq_word[topic][dictionary[word_id]] = sum_tfidf

    return topic_freq_word_id, topic_freq_word, topic_terms_20_above

def sort_frequency_word(topic_freq_word):

    print("# sort frequency word")
    topic_sorted = {}

    for topic, topic_words in topic_freq_word.items():
        topic_sorted[topic] = sorted(topic_words.items(), key=lambda x:x[1], reverse=True)

    return topic_sorted

def create_wordcloud(topic_freq_word, topic_terms_20_above, file_name, font_path, background_color="white", width=1024, height=674):

    print("# create word cloud")
    wordcloud_model = WordCloud(background_color=background_color, font_path=font_path, width=width, height=height)
    for topic, frequencies in topic_terms_20_above.items():
       wordcloud_obj = wordcloud_model.generate_from_frequencies(frequencies)
       save_name = "{0}{1}.png".format(file_name, topic)
       wordcloud_obj.to_file(save_name)

    print("# END")

if __name__ == "__main__":

    print("# start")
    docs, dictionary = create_gensim_dictionary("/home/tamashiro/AI/OPC/LDAPython/Data/対応方法", mecab_path=" -d /usr/lib/mecab/dic/mecab-ipadic-neologd")
    corpus, corpus_tfidf = create_gensim_corpus(docs, dictionary)
    for num_topics in range(6, 101):
        lda_model = "Data/model/model_topic{0}.lda".format(num_topics)
        cloud_name = "Data/wordcloud/wordcloud_{0}_topic".format(num_topics)
        corpus_lda_tfidf, topic_terms = lda(dictionary, corpus, corpus_tfidf, save_name=lda_model, num_topics=num_topics)
        #corpus_lda_tfidf, topic_terms = lda(dictionary, corpus, corpus_tfidf, lda_model=lda_model, num_topics=num_topics)
        docs_topic = doc2topic_id(corpus_lda_tfidf)
        _, topic_freq_word, topic_terms_20_above = word_cloud_list(dictionary, corpus_tfidf, docs_topic, topic_terms)
        create_wordcloud(topic_freq_word, topic_terms_20_above, cloud_name, "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")
