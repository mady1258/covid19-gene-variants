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

all_sequences_sampled="sequences_sampled"

for i in os.listdir(all_sequences_sampled):
        os.chdir(all_sequences_sampled)
        if i.endswith(".fasta"):
            os.chdir("..")
            ids,seqs=preprocessing.sequences(i)
            index=np.random.choice(ids.shape[0],1000,replace=False)
            ids_sampled=ids[index]
            seqs_sampled=seqs[index]
            os.chdir(all_sequences_sampled)
            f=open(i,'w')
            for j in range(len(ids_sampled)):
                f.write(">"+ids_sampled[j]+"\n")
                f.write(seqs_sampled[j]+"\n")
            os.chdir("..")

print("Time : ",total_time)
