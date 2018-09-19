from collections import Counter
import numpy as np
import random
import argparse


def gen(input, output, max_atm_length, num_exm):
    # define which character is an atom
    atoms = ['C', 'F', 'H', 'N', 'O', 'c', 'n', 'o']
    # save atom number of the word at the same time
    dict = {'words': [], 'atm-num': []}
    # load in words
    with open(input, 'rb') as f:
        for word in f:
            word = word.decode('utf-8').replace("\n", "")
            dict['words'].append(word)
            atm_num = 0
            counter = Counter(word)
            for atom in atoms:
                atm_num += counter[atom]
            dict['atm-num'].append(atm_num)
    # find the maximum atom number
    max_atm_num = max(dict['atm-num'])
    # separate the words into several bins according to atom number
    bins = {}
    for i in range(max_atm_num):
        bins[i + 1] = []
    for i in range(len(dict['words'])):
        bins[dict['atm-num'][i]].append(dict['words'][i])
    # maximum word number of one atom size of word
    max_num = []
    for i in range(max_atm_num):
        max_num.append(max_atm_length // (i + 1))
    # begin to generate sentences
    sents = []
    while len(sents) < num_exm:
        mole_num = [np.random.randint(max_num[i]) for i in range(len(max_num))]
        total_atm_num = 0
        for j in range(max_atm_num):
            total_atm_num += (j + 1) * mole_num[j]
        if total_atm_num > max_atm_length:
            continue
        else:
            total_mole_num = sum(mole_num)
            num_mole = 0
            sent = ''
            for j in range(max_atm_num):
                for k in range(mole_num[j]):
                    sent += random.sample(bins[j + 1], 1)[0]
                    num_mole += 1
                    if num_mole < total_mole_num:
                        sent += ' '
            sents.append(sent)
            print(len(sents))
    # save sentences
    text = '\n'.join(sents)
    path = output + 'txt'
    with open(path, 'ab+') as f:
        f.write(text.encode('utf-8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Text file previously tokenized '
                                      '(by whitespace) and preprocessed')
    parser.add_argument('output', help='save path, do not enter .txt')
    parser.add_argument('-n', help='number of examples',
                        dest='num_exm', default=10000, type=int)
    parser.add_argument('-a', help='maximum atom number in one sentence',
                        dest='max_atm_length', type=int, default=256)

    args = parser.parse_args()

    gen(args.input, args.output, args.max_atm_length, args.num_exm)

