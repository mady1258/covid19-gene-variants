import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import time
import os
from methods import methods
import warnings

warnings.filterwarnings("ignore")

t1=time.perf_counter()

sequence_vectors_dir="sequence_vectors_unaligned/"

all_vectors_file="all_vectors.txt"

labels_file="labels.txt"

f=open(all_vectors_file,'w')

f.close()

os.system("cat "+sequence_vectors_dir+"*.txt >> "+all_vectors_file)

#methods.novelty(all_vectors_file,labels_file,'B.1.1.7')

#methods.novelty(all_vectors_file,labels_file,'B.1.351')

methods.variant_lstm(all_vectors_file,labels_file,'B.1.351')

t2=time.perf_counter()

total_time=t2-t1

print("Time : ",total_time)
