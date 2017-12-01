#!/usr/bin/env python
import sys
from collections import OrderedDict
import numpy as np
import numpy.random as npr


def align_seqs_scores_1hot(seq_vecs, seq_scores):

    seq_headers = seq_vecs.keys()

    # construct lists of vectors
    train_scores = []
    train_seqs = []
    for header in seq_headers:
        train_seqs.append(seq_vecs[header])
        train_scores.append(seq_scores[header])

    # stack into matrices
    train_seqs = np.vstack(train_seqs)
    train_scores = np.vstack(train_scores)

    return train_seqs, train_scores


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


def hash_scores(scores_file):
    seq_scores = {}

    for line in open(scores_file):
        a = line.split()

        try:
            seq_scores[a[0]] = np.array([float(a[i]) for i in range(1,len(a))])
        except:
            print >> sys.stderr, 'Ignoring header line'

    # consider converting the scores to integers
    int_scores = True
    for header in seq_scores:
        if not np.equal(np.mod(seq_scores[header], 1), 0).all():
            int_scores = False
            break

    if int_scores:
        for header in seq_scores:
            seq_scores[header] = seq_scores[header].astype('int8')


    return seq_scores


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


def load_data_1hot(fasta_file, scores_file):
    # load sequences
    seq_vecs = hash_sequences_1hot(fasta_file)

    # load scores
    seq_scores = hash_scores(scores_file)

    # align and construct input matrix
    train_seqs, train_scores = align_seqs_scores_1hot(seq_vecs, seq_scores)

    return train_seqs, train_scores