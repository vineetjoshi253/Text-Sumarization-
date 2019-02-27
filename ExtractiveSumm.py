import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk.tokenize import sent_tokenize


df = pd.read_csv("Test.csv")


def Extract_Sentences(sentences):
    for s in df['article_text']:
        sentences.append(sent_tokenize(s))
        sentences = [y for x in sentences for y in x]
        
    return sentences  


def Extract_Word_Vectors(word_embeddings):
    f = open('glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    return word_embeddings

def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def Preprocessing(sentences):
    # Removing punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # Making alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    # Removing stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]  
    return clean_sentences
 
def Sentence_Vector(clean_sentences,word_embeddings):
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    return sentence_vectors

# Similarity matrix
def Similarity_Matrix(sentences,sentence_vectors):
    sim_mat = np.zeros([len(sentences), len(sentences)])
    print(len(sentences))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
    return sim_mat

#Converting matrix to graph
def Create_Graph(sim_mat):
    nx_graph = nx.from_numpy_array(sim_mat)
    return nx_graph

def Create_Summary(nx_graph,sentences):
    scores = nx.pagerank(nx_graph)
    #Summary
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    
    for i in range(1):
        print(ranked_sentences[i][1])
    
   

def main():
    print("Step1")
    sentences=[];
    sentences=Extract_Sentences(sentences)
    print("Step2")
    word_embeddings={}
    word_embeddings=Extract_Word_Vectors(word_embeddings)
    print("Step3")
    clean_sentences=Preprocessing(sentences)
    print("Step4")
    sentence_vectors=Sentence_Vector(clean_sentences,word_embeddings)
    print("Step5")
    sim_mat=Similarity_Matrix(sentences,sentence_vectors)
    print("Step6")
    nx_graph=Create_Graph(sim_mat)
    print("Step7")
    Create_Summary(nx_graph,sentences)
    

if __name__== "__main__":    
     main()