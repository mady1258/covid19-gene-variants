import os
import numpy as np
import codecs
from ast import literal_eval
from metrics import metric
from scipy import stats

def generator(cluster_path, cluster_dir, sequence_vectors_dir):
    date_clusters=[]
    for filename in sorted(os.listdir(cluster_dir)):
        if filename.endswith(".txt"):
            f=codecs.open(cluster_dir+filename,'r',encoding="ISO-8859-1")
            lines=f.readlines()
            clusters=[]
            for line in lines:
                cluster=literal_eval(line.strip())
                clusters.append(cluster)
            date_clusters.append(clusters)
    date_vectors=[]
    for filename in sorted(os.listdir(sequence_vectors_dir)):
        if filename.endswith(".txt"):
            f=codecs.open(sequence_vectors_dir+filename,'r',encoding="ISO-8859-1")
            lines=f.readlines()
            sequences=[]
            for line in lines:
                sequence=line.split("\t")[0]
                vector=line.split("\t")[1]
                vector=literal_eval(vector)
                sequences.append(sequence)
            date_vectors.append(vector)
    print(date_vectors)
    
