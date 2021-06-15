import os
import numpy as np
import codecs
from ast import literal_eval
from metrics import metric
from scipy import stats

def time_series(sequence_vectors_dir,cluster_dir,cluster_center_dir,bp_number): #returns single path starting at 0 index
    path_all=[]
    path=[]
    cluster_path=[]
    tmp=[]
    min_euclidean_distance=[]
    total_hamming_distance=[]
    date_sequences=[]
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
    for filename in sorted(os.listdir(sequence_vectors_dir)):
        if filename.endswith(".txt"):
            f=codecs.open(sequence_vectors_dir+filename,'r',encoding="ISO-8859-1")
            lines=f.readlines()
            sequences=[]
            #print(filename)
            for line in lines:
                sequence=line.split("\t")[0]
                sequences.append(sequence)
            date_sequences.append(sequences)
    cnt=0
    for filename in sorted(os.listdir(cluster_center_dir)):
        if filename.endswith(".txt"):
            #print(filename)
            f=open(cluster_center_dir+filename,'r',encoding="ISO-8859-1")
            lines=f.readlines()
            tmp=[]
            for line in lines:
                tmp.append(literal_eval(line))
            if cnt==0:
                for i in tmp:
                    path_all.append(i)
                path.append(path_all[0])
                cluster_path.append(0)
            else:
                s1=path[cnt-1]
                euclidean_distances=[]
                for s2 in tmp:
                    euclidean_distances.append(metric.euclidean_distance(s1,s2))
                path.append(tmp[np.argmin(euclidean_distances)]) #closest Euclidean distance forward path
                cluster_path.append(np.argmin(euclidean_distances))
                min_euclidean_distance.append(np.min(euclidean_distances))
            cnt=cnt+1
    date_cluster_sequences=[]
    for i in range(len(date_clusters)):
        cluster_sequences=[]
        for j in range(len(date_clusters[i])):
            if cluster_path[i]==date_clusters[i][j]:
                cluster_sequences.append(date_sequences[i][j])
        date_cluster_sequences.append(cluster_sequences)
    cnt2=1
    for k in range(1,len(date_cluster_sequences)):
        tmp=0
        cnt3=0
        for i in date_cluster_sequences[cnt2-1]:
            for j in date_cluster_sequences[cnt2]:
                tmp=tmp+metric.hamming_distance(i,j)
                cnt3=cnt3+1
        total_hamming_distance.append(tmp/cnt3)
        cnt2=cnt2+1
    #print(len(min_euclidean_distance))
    #print(len(total_hamming_distance))
    #print(stats.pearsonr(min_euclidean_distance, total_hamming_distance))
    cnt3=0
    backward_path_all=[]
    backward_path=[]
    backward_cluster_path=[]
    backward_min_euclidean_distance=[]
    for filename in sorted(os.listdir(cluster_center_dir),reverse=True):
        if filename.endswith(".txt"):
            #print(filename)
            f=open(cluster_center_dir+filename,'r',encoding="ISO-8859-1")
            lines=f.readlines()
            tmp=[]
            for line in lines:
                tmp.append(literal_eval(line))
            if cnt3==0:
                for i in tmp:
                    backward_path_all.append(i)
                backward_path.append(backward_path_all[bp_number])
                backward_cluster_path.append(bp_number)
            else:
                s1=path[cnt3-1]
                backward_euclidean_distances=[]
                for s2 in tmp:
                    backward_euclidean_distances.append(metric.euclidean_distance(s1,s2))
                backward_path.append(tmp[np.argmin(backward_euclidean_distances)]) #closest Euclidean distance backward path
                backward_cluster_path.append(np.argmin(backward_euclidean_distances))
                backward_min_euclidean_distance.append(np.min(backward_euclidean_distances))
            cnt3=cnt3+1
    backward_cluster_path.reverse()
    print(backward_cluster_path)
    #print(backward_cluster_path)
    #print(cluster_path)
    print(sum(1 for a,b in zip(backward_cluster_path,cluster_path) if a==b))
    return backward_cluster_path,cluster_path
