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

all_sequences="sequences"

methods.variant_lstm_binary(all_sequences)

t2=time.perf_counter()

total_time=t2-t1

print("Time : ",total_time)
