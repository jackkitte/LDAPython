# -*- coding: utf-8 -*-

import os
import pickle
import MeCab
import gensim
from wordcloud import WordCloud

def create_gensim_dictionary(data_path, mecab_path=None, no_below=2, no_above=0.1):

    if mecab_path is None:
        mecab = MeCab.Tagger("")
    else:
        mecab = MeCab.Tagger(mecab_path)

    for root, dirs, files in os.walk(data_path):
        print("# morphological analysis")
        docs = {}
        i = 1
        for docname in files:
            docs[docname] = []
            with open(os.path.join(data_path, docname), "r") as f:
                lines = f.readlines()
                for text in lines:
                    res = mecab.parseToNode(text)
                    while res:
                        arr = res.feature.split(",")
                        res = res.next
                        if arr[0] == "記号":
                            continue
                        elif len(arr[6]) == 1:
                            continue
                        else:
                            word = arr[6]
                            docs[docname].append(word)
            print("{0}.ファイル名：{1}".format(i, docname))
            i += 1

    dictionary = gensim.corpora.Dictionary(docs.values())
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    return docs, dictionary

def create_gensim_corpus(docs, dictionary):

    print("# create corpus")
    corpus = {}
    i = 1
    for file_name, word_list in docs.items():
        corpus[file_name] = dictionary.doc2bow(word_list)
        print("{0}.ファイル名：{1}".format(i, file_name))
        i += 1

    tfidf = gensim.models.TfidfModel(corpus.values())
    corpus_tfidf = tfidf[corpus.values()]

    return corpus, corpus_tfidf

def lda(dictionary, corpus, corpus_tfidf, lda_model=None, save_name="test.model", num_topics=2):

    print("# LDA")
    if lda_model is None:
        lda_model = gensim.models.LdaModel(corpus.values(), id2word=dictionary, num_topics=num_topics)
        lda_tfidf = lda_model[corpus_tfidf]
        lda_model.save(save_name)
    else:
        lda_model = gensim.models.LdaModel.load(lda_model)
        lda_tfidf = lda_model[corpus_tfidf]

    return lda_tfidf

def doc2topic_id(lda_tfidf):

    print("# doc2topic_id")
    corpus_topic = []
    for doc_id, doc in enumerate(lda_tfidf):
        corpus_topic.append(max(doc, key=lambda x:x[1])[0])

    return corpus_topic

def word_cloud_list(dictionary, corpus_tfidf, corpus_topic):

    print("# create word cloud list")
    topic_freq_word_id = {}
    topic_freq_word = {}
    corpus_tfidf_list = list(corpus_tfidf)
    
    for index, topic in enumerate(corpus_topic):
        for word_id, tfidf in corpus_tfidf_list[index]:
            if topic not in topic_freq_word_id.keys():
                topic_freq_word_id.update({topic: {}})
                topic_freq_word.update({topic: {}})
            if word_id not in topic_freq_word_id[topic]:
                topic_freq_word_id[topic].update({word_id: tfidf})
                topic_freq_word[topic].update({dictionary[word_id]: tfidf})
            else:
                sum_tfidf = topic_freq_word_id[topic][word_id] + tfidf
                topic_freq_word_id[topic][word_id] = sum_tfidf
                topic_freq_word[topic][dictionary[word_id]] = sum_tfidf

    return topic_freq_word_id, topic_freq_word

def sort_frequency_word(topic_freq_word):

    print("# sort frequency word")
    topic_sorted = {}

    for topic, topic_words in topic_freq_word.items():
        topic_sorted[topic] = sorted(topic_words.items(), key=lambda x:x[1], reverse=True)

    return topic_sorted

def create_wordcloud(topic_freq_word, file_name, font_path, background_color="white", width=1024, height=674):

    print("# cleate word cloud")
    wordcloud_model = WordCloud(background_color=background_color, font_path=font_path, width=width, height=height)
    for topic, frequencies in topic_freq_word.items():
       wordcloud_obj = wordcloud_model.generate_from_frequencies(frequencies)
       save_name = "{0}{1}.png".format(file_name, topic)
       wordcloud_obj.to_file(save_name)

    print("# END")

if __name__ == '__main__':

    print("# start")
    docs, dictionary = create_gensim_dictionary("./Data/対応方法", mecab_path=" -d /usr/lib64/mecab/dic/mecab-ipadic-neologd")
    corpus, corpus_tfidf = create_gensim_corpus(docs, dictionary)
    lda_tfidf = lda(dictionary, corpus, corpus_tfidf, lda_model="Data/model/model.lda")
    corpus_topic = doc2topic_id(lda_tfidf)
    _, topic_freq_word = word_cloud_list(dictionary, corpus_tfidf, corpus_topic)
    create_wordcloud(topic_freq_word, "wordcloudsample", "/usr/share/fonts/ipa-gothic/ipag.ttf")
