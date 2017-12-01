#!/usr/bin/env python
from optparse import OptionParser


def main():
    usage = 'usage: %prog [options] <data_file> <out_file>'
    parser = OptionParser(usage)
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide data file and an output prefix')
    else:
        data_file = args[0]
        out_file = args[1]

    data_file = open(data_file, 'r')
    stringArr = [line.strip() for line in data_file.readlines()]
    dataset = []
    for i in stringArr:
        dataset.append(float(i))
    dataout = []
    for i in dataset:
        if i >0.5:
            dataout.append('1')
        else:
            dataout.append('0')
    out_file = open(out_file, 'w')
    for i in dataout:
        out_file.write(i)
        out_file.write('\n')
    out_file.close()
    

			

if __name__ == '__main__':
    main()