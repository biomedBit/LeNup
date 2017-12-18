#!/usr/bin/env python


from optparse import OptionParser



def main():
    usage = 'usage: %prog [options] <positive_file> <negative_file> <seq_file> <list_file>'
    parser = OptionParser(usage)
    (options,args) = parser.parse_args()
    if len(args) != 4:
        parser.error('Must provide fasta file, targets file, and an output prefix')
    else:
        positive_file = args[0]
        negative_file = args[1]
        seq_file = args[2]
        list_file = args[3]


    positive_file = open(positive_file, 'r')
    line1 = positive_file.readlines()
    len1 = len(line1)
    str1 = [[0 for col in range(147)] for row in range(len1)]
    j = -1
    for line in line1:
        j += 1
        for i in range(147):
            str1[j][i] = line[i : i + 1]

            
    #negative_file = open(r'E:\nucl\negative_human_2300.txt','r')
    negative_file = open(negative_file,'r')
    line2 = negative_file.readlines()
    len2 = len(line2)
    str2 = [[0 for col in range(147)] for row in range(len2)]
    j = -1
    for line in line2:
        j += 1
        for i in range(147):
            str2[j][i] = line[i : i + 1]


    l = 1
    seq_file = open(seq_file, 'w')
    #seq_file=open('new1.txt', 'w')
    for i in range(len1):
        p = i * 147 + 1
        q = (i + 1) * 147
        seq_file.write('>chr:')
        seq_file.write(str(p))
        seq_file.write('-')
        seq_file.write(str(q))
        seq_file.write('(+)\n')
        l += 1
        for j in range(147):
            seq_file.write(str1[i][j])
        seq_file.write("\n")

    for i in range(len2):
        seq_file.write('>chr:')
        seq_file.write(str(((l - 1) * 147 + 1)))
        seq_file.write('-')
        seq_file.write(str((l * 147)))
        seq_file.write('(+)\n')
        l += 1
        for j in range(147):
            seq_file.write(str2[i][j])
        seq_file.write("\n")
    seq_file.close()


    
    str1 = ['1'for col in range(len1)]
    str2 = ['0'for col in range(len2)]
    k = 1

    list_file = open(list_file, 'w')
    #list_file = open('mewt.txt', 'w')
    for i in str1:
        list_file.write('chr:')
        list_file.write(str(((k - 1) * 147 + 1)))
        list_file.write('-')
        list_file.write(str((k * 147)))
        list_file.write('(+)\t')
        list_file.write(i)
        list_file.write("\n")
        k += 1
    for j in str2:
        list_file.write('chr:')
        list_file.write(str(((k - 1) * 147 + 1)))
        list_file.write('-')
        list_file.write(str((k * 147)))
        list_file.write('(+)\t')
        list_file.write(j)
        list_file.write("\n")
        k += 1
    list_file.close()
    
    
if __name__ == '__main__':
    main()