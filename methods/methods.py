#!/usr/bin/env python -W ignore::DeprecationWarning
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from ast import literal_eval
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD
from dataset import preprocessing
import warnings
from sklearn.utils import shuffle
from collections import Counter

def principal_components(all_vectors_file):
    warnings.filterwarnings("ignore")
    f=open(all_vectors_file,'r')
    lines=f.readlines()
    sequences=[]
    vectors=[]
    for line in lines: #get every sequence and vector
        sequence=line.split("\t")[0]
        vector=line.split("\t")[1]
        sequences.append(sequence)
        vector=literal_eval(vector)
        vectors.append(vector)
    f.close()
    scale=MinMaxScaler()
    vectors=scale.fit_transform(vectors)
    print(vectors)
    model=PCA(n_components=0.95,svd_solver='full') #PCA for 95% explained variance
    model.fit(vectors)
    pca_vectors=model.transform(vectors)
    return pca_vectors

def novelty(all_vectors_file,labels_file,variant_name):
    warnings.filterwarnings("ignore")
    f=open(all_vectors_file,'r')
    lines=f.readlines()
    sequences=[]
    vectors=[]
    for line in lines: #get every sequence and vector
        sequence=line.split("\t")[0]
        vector=line.split("\t")[1]
        sequences.append(sequence)
        vector=literal_eval(vector)
        vectors.append(vector)
    f.close()
    print("Before PCA")
    vectors=principal_components(all_vectors_file)
    print("After PCA")
    print(vectors.shape)
    f2=open(labels_file,'r')
    lines=f2.readlines()
    labels=[]
    for line in lines: #get every label
        label=line
        label=label.strip('\n')
        labels.append(label)
    f2.close()
    split=labels.index(variant_name)
    train_x=vectors[:split]
    train_y=labels[:split]
    test_x=vectors[split:]
    test_y=labels[split:]
    variant_test_pos=[]
    for i in range(len(test_y)):
        if test_y[i]==variant_name:
            variant_test_pos.append(i)
    model1=EllipticEnvelope(contamination=0,random_state=1)
    model1.fit(train_x)
    pred_y_model1=model1.predict(test_x)
    model2=IsolationForest(n_estimators=100, max_samples='auto',contamination=0, random_state=1)
    model2.fit(train_x)
    pred_y_model2=model2.predict(test_x)
    model3=LocalOutlierFactor(n_neighbors=20, algorithm='auto',metric='minkowski',contamination=0.01,novelty=True)
    model3.fit(train_x)
    pred_y_model3=model3.predict(test_x)
    model4=OneClassSVM(nu=0.01)
    model4.fit(train_x)
    pred_y_model4=model4.predict(test_x)
    cnt1=0
    cnt2=0
    cnt3=0
    cnt4=0
    total_variant_name=len(variant_test_pos)
    print(total_variant_name)
    print(len(vectors))
    print(len(labels))
    print(len(test_y))
    print(len(pred_y_model1))
    print(len(pred_y_model2))
    print(len(pred_y_model3))
    print(len(pred_y_model4))
    #print(variant_test_pos)
    for i in range(total_variant_name):
        if pred_y_model1[variant_test_pos[i]]==-1:    #detects novelty
            cnt1=cnt1+1
        if pred_y_model2[variant_test_pos[i]]==-1:    #detects novelty
            cnt2=cnt2+1
        if pred_y_model3[variant_test_pos[i]]==-1:    #detects novelty
            cnt3=cnt3+1
        if pred_y_model4[variant_test_pos[i]]==-1:    #detects novelty
            cnt4=cnt4+1
    print(cnt1/total_variant_name)
    print(cnt2/total_variant_name)
    print(cnt3/total_variant_name)
    print(cnt4/total_variant_name)

def kmers(seq,k=2):
    words=[seq[x:x+k].lower() for x in range(len(seq)-k+1)]
    sentence=' '.join(words)
    return sentence

def variant_lstm_binary(all_sequences,class_name):
    warnings.filterwarnings("ignore")
    X1=[]
    Y=[]
    cnt=0
    cnt3=0

    for i in os.listdir(all_sequences):
        os.chdir(all_sequences)
        if i.endswith(".fasta"):
            ids,seqs=preprocessing.sequences(i)
            index=np.random.choice(ids.shape[0],1000,replace=False)
            ids_sampled=ids[index]
            seqs_sampled=seqs[index]
            for j in seqs_sampled:
                X1.append(j[0:500]) 
                Y.append(cnt)
                print(cnt3)
                cnt3=cnt3+1
            cnt=cnt+1
        os.chdir("..")

    #min_seq_length=min(len(k) for k in X1)
    
    X1,Y=shuffle(X1,Y,random_state=0)
    Xunique=[]

    total_cnt=0

    for i in X1:
        k=kmers(i)
        X2=k.split(" ")
        for j in X2:
            if j not in Xunique:
                Xunique.append(j)
              

    X3=Xunique

    Xdict={}
    cnt=0
    for i in X3:
        Xdict[i]=cnt
        cnt=cnt+1

    vocab_size=cnt
    print("Vocabulary size:",vocab_size)

    X=[]
    X3=[]
    maxlen1=0
    for i in X1:
        k=kmers(i)
        X2=k.split(" ")
        X3=[]
        for j in X2:
            X3.append(Xdict[j])
        if len(X3)>=maxlen1:
                maxlen1=len(X3)
        X.append(np.array(X3))

    X=np.array(X)
    Y=np.array(Y)

    print("Maximum length:",maxlen1)

    X=pad_sequences(X, maxlen=maxlen1)

    trainlen=round(0.8*len(X))

    X_train=X[:trainlen]
    Y_train=Y[:trainlen]
    X_test=X[trainlen:]
    Y_test=Y[trainlen:]

    print(X_train.shape)

    embedding_vector=50
    model=Sequential()
    model.add(Embedding(vocab_size, embedding_vector, input_length=maxlen1))
    model.add(LSTM(10))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=256)
    scores=model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

def variant_lstm_multiclass(all_sequences):
    warnings.filterwarnings("ignore")
    X1=[]
    Y=[]
    cnt=0

    for i in os.listdir(all_sequences):
        os.chdir(all_sequences)
        if i.endswith(".fasta"):
            ids,seqs=preprocessing.sequences(i)
            index=np.random.choice(ids.shape[0],1000,replace=False)
            ids_sampled=ids[index]
            seqs_sampled=seqs[index]
            for j in seqs_sampled:
                X1.append(j[0:500]) 
                Y.append(cnt)
            cnt=cnt+1
        os.chdir("..")

    X1,Y=shuffle(X1,Y,random_state=0)
    Xunique=[]

    cnt3=0

    for i in X1:
        print(cnt3)
        k=kmers(i)
        X2=k.split(" ")
        for j in X2:
            if j not in Xunique:
                Xunique.append(j)
        cnt3=cnt3+1

    X3=Xunique
    
    Xdict={}
    cnt=0
    for i in X3:
        Xdict[i]=cnt
        cnt=cnt+1

    vocab_size=cnt
    print("Vocabulary size:",vocab_size)

    X=[]
    X3=[]
    maxlen1=0
    for i in X1:
        k=kmers(i)
        X2=k.split(" ")
        X3=[]
        for j in X2:
            X3.append(Xdict[j])
        if len(X3)>=maxlen1:
            maxlen1=len(X3)
        X.append(np.array(X3))

    X=np.array(X)
    Y=np.array(Y)

    print("Maximum length:",maxlen1)

    X=pad_sequences(X, maxlen=maxlen1)

    trainlen=round(0.8*len(X))

    X_train=X[:trainlen]
    Y_train=Y[:trainlen]
    X_test=X[trainlen:]
    Y_test=Y[trainlen:]

    print(X_train.shape)

    embedding_vector=50
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector, input_length=maxlen1))
    model.add(LSTM(10))
    model.add(Dense(20, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=256)
    scores=model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
