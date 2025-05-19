# import nltk 
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# from nltk.cluster.util import cosine_distance
# import numpy as np
# import networkx as nx 


# def read_article(file_name):
#     file =open(file_name,"r")
#     filedata = file.readlines()
#     article = filedata[0].split(". ")
#     sentences = []
#     for sentence in article:
#         sentences.append(sentence.replace("[^a-zA-Z]"," ").split(" "))
#     sentences.pop()
#     return sentences    

# def sentence_similarity(sent1,sent2,stopwords=None):
#     if stopwords is None:
#         stopwords=[]
#     sent1 = [w.lower() for w in sent1]
#     sent2 = [w.lower() for w in sent2]
#     all_words = list(set(sent1+sent2))
    
#     vector1= [0]*len(all_words)
#     vector2 =  [0]*len(all_words)
    
#     for w in sent1:
#         if w in stopwords:
#             continue
#         vector1[all_words.index(w)]+=1
#     for w in sent2:
#         if w in stopwords:
#             continue
#         vector2[all_words.index(w)]+=1
#     return 1-cosine_distance(vector1,vector2)


# def gen_sim_matrix(sentences,stop_words):
#     similarity_matrix=np.zeros((len(sentences),len(sentences)))
#     for idx1 in range(len(sentences)):
#         for idx2 in range(len(sentences)):
#             if idx1==idx2:
#                 continue
#             similarity_matrix[idx1][idx2]=sentence_similarity(sentences[idx1],sentences[idx2],stop_words)
#     return similarity_matrix


# def generate_summary(file_name,top_n=5):
#     stop_words=stopwords.words('english')
#     summarizer_text=[]
#     sentences = read_article(file_name)
#     sentences_similarity_matrix = gen_sim_matrix(sentences,stop_words)
#     sentences_similarity_graph = nx.from_numpy_array(sentences_similarity_matrix)
#     scores = nx.pagerank(sentences_similarity_graph)
#     ranked_sentence=sorted(((scores[i],s)for i,s in enumerate(sentences)),reverse=True)
#     for i  in range(top_n):
#         summarizer_text.append(" ".join(ranked_sentence[i][1]))
#     print("summary \n",". ".join(summarizer_text))
    
# generate_summary("ml.txt",2)

import nltk
import re
import numpy as np
import networkx as nx
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance

# Download stopwords (only needed once)
nltk.download('stopwords')

def read_article(file_name):
    with open(file_name, "r") as file:
        filedata = file.read()
    article = filedata.split(". ")
    sentences = []
    for sentence in article:
        # Clean the sentence using regex
        cleaned_sentence = re.sub(r"[^a-zA-Z]", " ", sentence)
        words = cleaned_sentence.split()
        sentences.append(words)
    return sentences

def sentence_similarity(sent1, sent2, stop_words=None):
    if stop_words is None:
        stop_words = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        if w in stop_words:
            continue
        vector1[all_words.index(w)] += 1

    for w in sent2:
        if w in stop_words:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

def gen_sim_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarizer_text = []

    sentences = read_article(file_name)

    # Build similarity matrix
    similarity_matrix = gen_sim_matrix(sentences, stop_words)

    # Create graph and apply PageRank
    similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(similarity_graph)

    # Rank sentences
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # Select top N sentences
    for i in range(top_n):
        summarizer_text.append(" ".join(ranked_sentences[i][1]))

    # Print the summary
    print("Summary:\n")
    print(". ".join(summarizer_text))

# Example usage
generate_summary("ml.txt", 5)
