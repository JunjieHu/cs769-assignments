# Assignment 1
by Junjie Hu

This is an exercise in developing a text classifier based on neural network toolkit for NLP, part of University of Wisconsin-Madison's [CS 769: Advanced NLP](https://junjiehu.github.io/cs769-spring22/).

The most important files it contains are the following:
1. **model.py:** This is what you'll need to implement. It implements a very basic version of a neural network model using [PyTorch](https://github.com/pytorch/pytorch). Some code is provided, but important functionality is not included. Please implement [Deep Averaging Network](https://www.aclweb.org/anthology/P15-1162.pdf) for text classification. You can feel free to make any modifications to make it a better model. *However,* the original version of `DanModel` will also be tested and results will be used in grading, so the original `DanModel` must also run with your `model.py` implementation.
2. **main.py:** training code for text classification task.
3. **setup.py:** this is blank, but if your classifier implementation needs to do some sort of data downloading (e.g. of pre-trained word embeddings) you can implement this here. It will be run before running your implementation of model.py.
4. **data/:** Two datasets, one from the Stanford Sentiment Treebank with tree info removed and another from IMDb reviews. **The labels in cfimdb-test.txt are replaced by 0** for blind testing and grading.

## Assignment Details

Important Notes:
- There is a detailed description of the [code structure](#code-structure) below, including a description of which parts you will need to implement. 
- The only allowed external library is `numpy` and `pytorch`, no other external libraries are allowed. As the datasets are small, a DAN model with a similar size in their origial paper can be trained within a few minutes (<30 minutes) using CPU, while it's also encouraged to train more advanced models that may require GPU. Please check available resources such as [CHTC](https://chtc.cs.wisc.edu/) or [Google's Colab](https://colab.research.google.com/).
- We will run your code with the following commands (i.e., `run_exp.sh`) using both the original `main.py` and your updated `model.py` if you make any modifications there. Because of this, make sure that whichever setting you think is best is reproducible using exactly these commands (where you replace `CAMPUSID` with your 10-digit campus ID):
    - `CAMPUSID="9xx1234567"`
    - `mkdir -p CAMPUSID`
    - `python main.py --train=data/sst-train.txt --dev=data/sst-dev.txt --test=data/sst-test.txt --dev_out=$CAMPUSID/sst-dev-output.txt --test_out=$CAMPUSID/sst-test-output.txt`
    - `python main.py --train=data/cfimdb-train.txt --dev=data/cfimdb-dev.txt --test=data/cfimdb-test.txt --dev_out=CAMPUSID/cfimdb-dev-output.txt --test_out=CAMPUSID/cfimdb-test-output.txt`
- Please remember to set your default hyper-parameter settings in your own `run_exp.sh` since we will also run your experiment by `bash run_exp.sh` (without extra arguments).
- Reference accuracies: If you implement things exactly in our way and use the default hyper-parameters and use the same environment (python 3.8 + numpy 1.21.1 + pytorch 1.10.2), you may get the accuracies of dev=0.3951, test=0.4122, and on cfimdb dev=0.9224.

The submission file should be a zip file with the following structure (assuming the campus id is `CAMPUSID`):

- CAMPUSID/
- CAMPUSID/main.py `# completed main.py (i.e., `pad_sentences` function)`
- CAMPUSID/model.py `# completed model.py with any of your modifications`
- CAMPUSID/vocab.py `# no modification needed`
- CAMPUSID/setup.py `# only if you need to set up anything else`
- CAMPUSID/sst-dev-output.txt `# output of the dev set for SST data`
- CAMPUSID/sst-test-output.txt `# output of the test set for SST data`
- CAMPUSID/cfimdb-dev-output.txt `# output of the dev set for CFIMDB data`
- CAMPUSID/cfimdb-test-output.txt `# output of the test set for CFIMDB data`
- CAMPUSID/report.pdf `# (optional), report. here you can describe anything particularly new or interesting that you did`
- CAMPUSID/README `# (optional) only if you use pre-trained word vectors such as GloVE and FastText. Do not upload the word embedding file. Instead, mention in the README with a download link to the word embedding file that you use for "--emb_file" in main.py.`

Grading information:
- **100:** Submissions that implement something new and achieve particularly large accuracy improvements (for SST, this would be on the order of 2\% over the baseline, but you must also achieve gains on IMDB)
- **95:** You additionally implement something else on top of the missing pieces, some examples include:
    - Changing the training procedure such as mini-batching, optimizer, early-stop, learning rate scheduling.
    - Incorporating pre-trained word embeddings, such as those from [fasttext](https://fasttext.cc/) or [GloVe](https://nlp.stanford.edu/projects/glove/)
    - Changing the model architecture significantly
- **90:** You implement all the missing pieces and the original `model.py` code achieves comparable accuracy to our reference implementation (about 40% on SST)
- **85:** All missing pieces are implemented, but accuracy is not comparable to the reference.
- **80 or below:** Some parts of the missing pieces are not implemented.

Tools:
- `prepare_submit.py` can help to create(1) or check(2) the to-be-submitted zip file. It will throw assertion errors if the format is not expected, and we will *not accept submissions that fail this check*. Usage: (1) To create and check a zip file with your outputs, run `python3 prepare_submit.py path/to/your/output/dir CAMPUSID`, (2) To check your zip file, run `python3 prepare_submit.py path/to/your/submit/zip/file.zip CAMPUSID`

## Code Structure
Here is a walk-through of the main components in this repo. Note that some functions are not completely implemented, where **to-be-implemented** parts will `raise NotImplementedError()`. 

### [model.py](model.py)
This file contains a `BaseModel` class and a `DanModel` class. Here are the **to-be-implemented** parts:
- **define_model_parameters()**: Define the model's parameters such as embedding layer, feedforward layer, activation function (ReLU, or others). See [PyTorch API](https://pytorch.org/docs/stable/nn.html) for different layers.
- **init_model_parameters()**: Initialize the model's parameters using uniform initialization within a range or other methods. See `Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010)` for details about Xavier/Glorot initialization, and this blog for [more details about initialization in general](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79).
- **load_embedding()**: Read a file line by line to contruct a word embedding matrix (numpy.array) for words appearing in `vocab`.
- **copy_embedding_from_numpy**: Copy the word embeddings (numpy.array) to the PyTorch's embedding matrix.

### [main.py](main.py)
This file contains the training and evaluation functions for learning a text classifier. Only one function is **to-be-implemented**:
- **pad_sentences()**: Given a mini-batch of sentences (i.e., a list of a list of word ids) with different length (e.g., `[[1,2,5], [3,4], [4,6,8,9]]` denotes a mini-batch of 3 sentences in which their word lengths are different`|s_1|=3, |s_2|=2, |s_3|=4`), find the maximum word sequence length (i.e., `max_seq_length=4`), and add the pad ids to the end of sentences to make a mini-batch of size `[batch_size, max_seq_length]` (e.g., `[3,4]` in this example). 

### [vocab.py](vocab.py)
This file reads a list of tokenized sentences and contruct the vocabulary for words. We can reuse this to contruct the vocabulary for tags (e.g., `Positive` and `Negative` in a sentiment classification task). 

## References

Stanford Sentiment Treebank: https://www.aclweb.org/anthology/D13-1170.pdf

IMDb Reviews: https://openreview.net/pdf?id=Sklgs0NFvr
