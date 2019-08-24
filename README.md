Recurrent-Autoencoder
=====================
This program is an auto-encoder with RNN. It is a kind of unsupervised learning, what we want is the latent vector produced by the encoder. We can use it to transform a variational length sequence into a fixed length vector. Specifically, we want to do that for SMILES code which is a chemical expression of molecules.

Environment
-----------
* python2 or python3
* tensorflow
* progressbar

Prepare data
-------------
1. There is a __dictionary.txt__ with 256 words in it. You can use __gen-moles.py__ to generate sentences based on the words in the library. Use `python gen-moles.py -h` for more information about how to use it. You can change maximum atom length and number of sentences if you want. And it gives back a '.txt' file which wil be the input file of our neural net.
Sentences are like this:
> C C C N C O N O C N C C N N O O N N C C C O N N N N N C N O O C N C CO CO C#C C#N CO C=O C#C C#C CO C#N C=O C#C C#C CO C=O C#C C1CC1 C1CC1 CC#C CCC CC#N OCC=O COC(=O)N CCOC=O CC(=NO)C NC(=N)C#N OC1COC1 N=CNC=O O=CNC=O CCCOC CCC#CC o1nnnn1 CC(C)C#C FC(F)(F)F OC1COC1 o1nnnn1 NC(=O)C#C Nc1cocc1 CC(C)(C)C=O C/C(=N\O)/C#C [nH]1ccnc1 O[C@H]1C[C@H]1O O=c1[nH]cc[nH]1 CC(=O)O[CH][NH] [NH][C@H]1OC=CO1 [NH][C@@H]1NC=CO1
2. Now use the __prepare-data.py__ to translate the input file into machine learning data. Use `python prepare-data.py -h` for more information about how to use it. The output will be in the directory you gave it, there are training data, validation data and the dictionary.

Train it
--------
Basically all the files related with the neural net are in __neural-net__ directory. Use __train-autoencoder.py__ for training. you can adjust parameters if you want. Use `python train-autoencoder.py -h` for more information about how to use it.

Test it
-------
1. Use __interactive.py__ to test if the neural net can give the same thing as your input. After opening this file, you can see a interactive environment, you can type in your sentence, after "in:" shows up. And the result will automatically shows up after "ou:". Use `python interactive.py -h`
2. Use __codify-sentences.py__ to translate molecule sentences into vectors. You need a file contains several molecule sentences in it. And it will give you a '.npy' file which contains state arrays in it.

Reference
---------
https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
