import random
import argparse

def create_data(num_vocabs, max_length, num_examples):
    sents = []
    vocab = random.sample(range(1, 133886), num_vocabs)
    for i in range(0, num_examples):
        length = random.randint(2, max_length)
        sentence = random.sample(vocab, length)
        sentence = [str(item) for item in sentence]
        sentence = ' '.join(sentence)
        sents.append(sentence)
    return sents

if __name__ == '__main__':
    class_ = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=class_)
    parser.add_argument('-v', help='vocabulary number',
                        default=100, dest='num_vocabs', type=int)
    parser.add_argument('-m', help='maximum length of the sentence',
                        default=10, dest='max_length', type=int)
    parser.add_argument('-n', help='example number',
                        default=10000, dest='num_examples', type=int)
    parser.add_argument('output', help='output file name')
    args = parser.parse_args()
    sents = create_data(args.num_vocabs, args.max_length, args.num_examples)
    text = '\n'.join(sents)
    path = args.output + '.txt'
    with open(path, 'wb') as f:
        f.write(text.encode('utf-8'))




