#!/usr/bin/env python
from optparse import OptionParser
import h5py
import numpy.random as npr
import numpy as np
import dna_io_test


def main():
    usage = 'usage: %prog [options] <fasta_file>'
    parser = OptionParser(usage)
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide fasta file, targets file, and an output prefix')
    else:
        fasta_file = args[0]

    seqs = dna_io_test.load_data_1hot(fasta_file)

    seqs = seqs.reshape((seqs.shape[0],4,1,seqs.shape[1]/4))
    name = 'dmetestdata.h5'
    h5f = h5py.File(name, 'w')
    h5f.create_dataset('test_in', data=seqs)
    h5f.close()
	

if __name__ == '__main__':
    main()
