import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import time
import os
import warnings
from dataset import preprocessing

warnings.filterwarnings("ignore")

t1=time.perf_counter()

all_sequences="sequences_sampled_binary"

for i in os.listdir(all_sequences):
        if os.path.isdir(all_sequences+"/"+i):
                os.chdir(all_sequences+"/"+i)
                print(i)
                print(os.getcwd())
                for j in os.listdir(os.getcwd()):
                    print(os.getcwd())
                    if j=="other.fasta":
                        ids,seqs=preprocessing.sequences(j)
                        index=np.random.choice(ids.shape[0],1000,replace=False)
                        ids_sampled=ids[index]
                        seqs_sampled=seqs[index]
                        f=open("others.fasta",'w')
                        for k in range(len(ids_sampled)):
                            f.write(">"+ids_sampled[k]+"\n")
                            f.write(seqs_sampled[k]+"\n")
                os.chdir("..")
                os.chdir("..")

t2=time.perf_counter()

total_time=t2-t1

print("Time : ",total_time)

