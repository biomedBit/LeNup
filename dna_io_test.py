#!/usr/bin/env python
import sys
from collections import OrderedDict
import numpy as np
import numpy.random as npr


def align_seqs_scores_1hot(seq_vecs):

    seq_headers = seq_vecs.keys()

    # construct lists of vectors
    train_seqs = []
    for header in seq_headers:
        train_seqs.append(seq_vecs[header])

    # stack into matrices
    train_seqs = np.vstack(train_seqs)

    return train_seqs


def dna_one_hot(seq, seq_len=None, flatten=True):
    if seq_len == None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq)-seq_len)/2
            seq = seq[seq_trim:seq_trim+seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len-len(seq))/2

    seq = seq.upper()

    seq = seq.replace('A','0')
    seq = seq.replace('C','1')
    seq = seq.replace('G','2')
    seq = seq.replace('T','3')

    # map nt's to a matrix 4 x len(seq) of 0's and 1's.
    #  dtype='int8' fails for N's
    seq_code = np.zeros((4,seq_len), dtype='float16')
    for i in range(seq_len):
        if int(seq[i-seq_start]) == 5:
            aaa = 1
        else:
            seq_code[int(seq[i-seq_start]),i] = 1
    # flatten and make a column vector 1 x len(seq)
    if flatten:
        seq_vec = seq_code.flatten()[None,:]
    return seq_vec



def hash_sequences_1hot(fasta_file,):

    seq_len = 0
    seq = ''
    for line in open(fasta_file):
        if line[0] == '>':
            if seq:
                seq_len = max(seq_len, len(seq))

            header = line[1:].rstrip()
            seq = ''
        else:
            seq += line.rstrip()

    if seq:
        seq_len = max(seq_len, len(seq))

    # load and code sequences
    seq_vecs = OrderedDict()
    seq = ''
    for line in open(fasta_file):
        if line[0] == '>':
            if seq:
                seq_vecs[header] = dna_one_hot(seq, seq_len)

            header = line[1:].rstrip()
            seq = ''
        else:
            seq += line.rstrip()

    if seq:
        seq_vecs[header] = dna_one_hot(seq, seq_len)

    return seq_vecs


def load_data_1hot(fasta_file):
    # load sequences
    seq_vecs = hash_sequences_1hot(fasta_file)

    # align and construct input matrix
    train_seqs = align_seqs_scores_1hot(seq_vecs)

    return train_seqs