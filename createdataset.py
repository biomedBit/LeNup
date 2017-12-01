#!/usr/bin/env python
from optparse import OptionParser
import h5py
import numpy.random as npr
import numpy as np
import dna_io
    
def main():
    usage = 'usage: %prog [options] <fasta_file> <targets_file> <out_file>'
    parser = OptionParser(usage)
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide fasta file, targets file, and an output prefix')
    else:
        fasta_file = args[0]
        targets_file = args[1]
        out_file = args[2]

    seqs, targets = dna_io.load_data_1hot(fasta_file, targets_file)

    seqs = seqs.reshape((seqs.shape[0],4,1,seqs.shape[1]/4))

    target_labels = open(targets_file).readline().strip().split('\t')


    order = npr.permutation(seqs.shape[0])
    seqs = seqs[order]
    targets = targets[order]

    order = npr.permutation(seqs.shape[0])
    seqs = seqs[order]
    targets = targets[order]
    
    seqsnum1= int(seqs.shape[0] / 20)
    x = 0
	
    kseqs = []
    ktargets = []
    for i in range(19):
        kseqs.append(seqs[x:x + seqsnum1,:].tolist())
        ktargets.append(targets[x:x + seqsnum1,:].tolist())
        x += seqsnum1
	
    kseqs.append(seqs[x:seqs.shape[0],:].tolist())
    ktargets.append(targets[x:seqs.shape[0],:].tolist())

    length = len(kseqs[0])

    for i in range(20):
        name = out_file + str(i + 1) + '.h5'
        valid_seqs = kseqs[i]
        valid_targets = ktargets[i]
        if i != 0:
            train_seqs = kseqs[0][:][:][:][:]
            train_targets = ktargets[0][:][:][:][:]
        else:
            train_seqs = kseqs[1][:][:][:][:]
            train_targets = ktargets[1][:][:][:][:]
        for j in range(1, 20):
            if (j != i) and (i != 0):
                train_seqs += kseqs[j][:][:][:][:]
                train_targets += ktargets[j][:][:][:][:]
            elif (j != 1) and (i == 0):
                train_seqs += kseqs[j][:][:][:][:]
                train_targets += ktargets[j][:][:][:][:]
        h5f = h5py.File(name, 'w')
        h5f.create_dataset('target_labels', data=target_labels)
        h5f.create_dataset('train_in', data=train_seqs)
        h5f.create_dataset('train_out', data=train_targets)
        h5f.create_dataset('valid_in', data=valid_seqs)
        h5f.create_dataset('valid_out', data=valid_targets)
        h5f.close()

			

if __name__ == '__main__':
    main()