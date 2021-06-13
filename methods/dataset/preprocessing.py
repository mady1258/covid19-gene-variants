import pandas as pd
from operator import add
from Bio import SeqIO
import collections

def select_s_proteins(filename):
    df=pd.read_csv(filename)
    df=df.loc[df['Protein']=='surface glycoprotein']
    df=df.loc[df['Nuc_Completeness']=='complete']
    df=df.loc[df['Host']=='Homo sapiens']
    return df

def select_genes(filename,clade):
    df=pd.read_csv(filename,sep='\t')
    df=df.loc[df['Clade']==clade]
    return df

def sequences(filename):
    ids=[]
    seqs=[]
    for record in SeqIO.parse(filename,"fasta"):
        ids.append(record.id)
        seqs.append(str(record.seq))
    counter=collections.Counter(seqs)
    return ids,seqs

def prot_vec(filename):
    df=pd.read_csv(filename,sep='\t')
    cols=len(df.columns)
    df1=df.iloc[:,0:1]
    df2=df.iloc[:,1:cols]
    return df1.values,df2.values

def sequence_to_vec(sequence,proteins,vectors):
    sequence_vector=[]
    flag=0
    for i in range(len(proteins)):
        sequence_vector.append(0)
    for i in range(len(sequence)-2):
        if sequence[i]=='-' or sequence[i+1]=='-' or sequence[i+2]=='-':
            protein_sequence='<unk>'
        else:
            protein_sequence=sequence[i]+sequence[i+1]+sequence[i+2]
        for j in range(len(proteins)):
            if protein_sequence==proteins[j][0]:
                sequence_vector=list(map(add,sequence_vector,vectors[j]))
    return sequence_vector

def sequence_to_int_vec(sequence):
    seq_str=""
    flag=0
    for i in range(len(sequence)-2):
        protein_sequence=sequence[i]+sequence[i+1]+sequence[i+2]
        seq_str=seq_str+" "+protein_sequence
    return seq_str

