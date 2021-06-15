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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD
from dataset import preprocessing
import warnings
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from collections import Counter

def kmers(seq,k=3):
    words=[seq[x:x+k].lower() for x in range(len(seq)-k+1)]
    sentence=' '.join(words)
    return sentence

def variant_lstm_binary(all_sequences):
    f=open(all_sequences+"_output.txt",'w')
    f.close()
    positional_cnt=0
    total_prediction=np.zeros((400,1))
    for positional in range(51):
        f=open(all_sequences+"_output.txt",'a')
        print("positional_cnt", positional_cnt, positional_cnt+500)
        warnings.filterwarnings("ignore")
        X1=[]
        Y=[]
        cnt=0
        cnt3=0

        for i in os.listdir(all_sequences):
            os.chdir(all_sequences)
            if i.endswith(".fasta"):
                ids,seqs=preprocessing.sequences(i)
                for j in seqs:
                    X1.append(j[positional_cnt:positional_cnt+500])
                    Y.append(cnt)
                    cnt3=cnt3+1
                cnt=cnt+1
            os.chdir("..")
        
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

        embedding_vector=100
        model=Sequential()
        model.add(Embedding(vocab_size, embedding_vector, input_length=maxlen1))
        model.add(LSTM(10))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #print(model.summary())
        model.fit(X_train, Y_train, epochs=50, batch_size=128, verbose=0)
        Y_predict=model.predict(X_test)
        total_prediction=np.add(total_prediction,Y_predict)
        scores=model.evaluate(X_test, Y_test, verbose=0)
        positional_cnt=positional_cnt+500
        f.write(str(round(scores[1]*100,2))+"\n")
        f.close()
    total_prediction=total_prediction/51
    total_prediction=(total_prediction>0.5).astype(int)
    total_accuracy=accuracy_score(Y_test,total_prediction)
    f=open(all_sequences+"_output.txt",'a')
    f.write(str(round(total_accuracy*100,2)))
    print("Total Accuracy: %.2f%%" % (total_accuracy*100))
        

def variant_lstm_multiclass(all_sequences):
    positional_cnt=0
    for positional in range(51):
        print("positional_cnt", positional_cnt, positional_cnt+500)
        warnings.filterwarnings("ignore")
        X1=[]
        Y=[]
        cnt=0

        for i in os.listdir(all_sequences):
            os.chdir(all_sequences)
            if i.endswith(".fasta"):
                ids,seqs=preprocessing.sequences(i)
                for j in seqs:
                    X1.append(j[positional_cnt:positional_cnt+500]) 
                    Y.append(cnt)
                cnt=cnt+1
            os.chdir("..")

        X1,Y=shuffle(X1,Y,random_state=0)
        Xunique=[]

        cnt3=0

        for i in X1:
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
        model.fit(X_train, Y_train, epochs=50, batch_size=128)
        Y_predict=model.predict(X_test)
        print(Y_predict)
        scores=model.evaluate(X_test, Y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        positional_cnt=positional_cnt+500+1
