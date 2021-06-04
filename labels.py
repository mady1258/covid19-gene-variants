from dataset import preprocessing
import collections
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import time
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import datetime

t1=time.perf_counter()

ids,seqs=preprocessing.sequences('spikeprot0601/spikeprot0601.fasta') #all spike proteins

proteins,vectors=preprocessing.prot_vec('protVec_100d_3grams.csv') #proteins and vectors
    
df3=pd.read_csv('nextstrain_ncov_global_metadata.tsv',sep='\t') #subset of spike proteins

unaligned_sequences="unaligned_sequences.fasta"

label_file='labels.txt'

dates_all=[]
ids_all=[]
seqs_all=[]

f=open(label_file,'w')
f.close()

cnt3=0
for i in range(len(ids)):
    try:
        dates_all.append(ids[i].split("|")[2]) #collecting all dates
        ids_all.append(ids[i].split("|")[3]) #collecting all IDs
        seqs_all.append(seqs[i]) #collecting all sequences
        cnt3=cnt3+1
    except:
        continue

cnt1=0

dates_subset=[]
ids_subset=[]

variants=df3['PANGO Lineage'].values
variants_subset=[]

print(variants)

for i in range(len(ids_all)):
    if ids_all[i] in df3['gisaid_epi_isl'].values:
        ids_subset.append(ids_all[i])
        dates_subset.append(dates_all[i])
        cnt1=cnt1+1

print(cnt1)


variants_date=df3['Collection Data'].values
variants_date_subset=[]

for i in range(len(df3['gisaid_epi_isl'].values)):
    if df3['gisaid_epi_isl'].values[i] in ids_all:       
        variants_subset.append(variants[i])
        variants_date_subset.append(variants_date[i])

date_sequence_vectors=[]
seq_sequence_vectors=[]
sequence_vectors=[]

for i in range(len(ids_subset)):
    if ids_subset[i] in df3['gisaid_epi_isl'].values:
        sequence_vectors.append(0)
        date_sequence_vectors.append(dates_subset[i]) #collecting corresponding date
        seq_sequence_vectors.append(seqs_all[i]) #collecting corresponding seqs

dates_unique=list(df3['Collection Data'].unique()) #collecting unique dates for clustering

flag=0
dates_rest=[]

print("date sequence vector length", len(date_sequence_vectors))

for i in range(len(dates_unique)):
    print(dates_unique[i])
    flag=0
    for j in range(len(date_sequence_vectors)):
        if date_sequence_vectors[j]==datetime.datetime.strptime(dates_unique[i],'%m/%d/%y').strftime('%Y-%m-%d'):
            flag=1
    if flag==1:
        if flag==0:
            dates_rest.append(dates_unique[i])

cnt=0

for i in range(len(variants_subset)):
    if variants_date_subset[i] not in dates_rest:
        cnt=cnt+1
        f=open(label_file,'a')
        f.write(str(variants_subset[i])+"\n")
        f.close()

t2=time.perf_counter()

total_time=t2-t1

print("Time : ",total_time)

print(cnt)
