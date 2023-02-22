# minbert Assignment
by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt and Brendon Boldt

This is an exercise in developing a minimalist version of BERT, adapted from CMU's [CS11-747: Neural Networks for NLP](http://www.phontron.com/class/nn4nlp2020/).

In this assignment, you will implement some important components of the BERT model to better understanding its architecture. 
You will then perform sentence classification on ``sst`` dataset and ``cfimdb`` dataset with the BERT model.

## Assignment Details

### Important Notes
* Follow `setup.sh` to properly setup the environment and install dependencies.
* There is a detailed description of the code structure in [structure.md](./structure.md), including a description of which parts you will need to implement.
* You are only allowed to use `torch`, no other external libraries are allowed (e.g., `transformers`).
* We will run your code with the following commands, so make sure that whatever your best results are reproducible using these commands (where you replace CAMPUSID with your campus ID):
```
mkdir -p CAMPUSID

python3 classifier.py --option [pretrain/finetune] --epochs NUM_EPOCHS --lr LR --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt
```
## Reference accuracies: 

Pretraining for SST:
Dev Accuracy: 0.391 (0.007)
Test Accuracy: 0.403 (0.008)

Mean reference accuracies over 10 random seeds with their standard deviation shown in brackets.

Finetuning for SST:
Dev Accuracy: 0.515 (0.004)
Test Accuracy: 0.526 (0.008)

Finetuning for CFIMDB:
Dev Accuracy: 0.966 (0.007)
Test Accuracy: -

### Submission
The submission file should be a zip file with the following structure (assuming the campus id is ``CAMPUSID``):
```
CAMPUSID/
├── base_bert.py
├── bert.py
├── classifier.py
├── config.py
├── optimizer.py
├── sanity_check.py
├── tokenizer.py
├── utils.py
├── README.md
├── structure.md
├── sanity_check.data
├── sst-dev-output.txt 
├── sst-test-output.txt 
├── sst-train-log.txt 
├── cfimdb-dev-output.txt 
├── cfimdb-test-output.txt 
├── cfimdb-train-log.txt 
├── run_exp.sh 
└── setup.py
```

`prepare_submit.py` can help to create(1) or check(2) the to-be-submitted zip file. It will throw assertion errors if the format is not expected, and we will *not accept submissions that fail this check*. Usage: (1) To create and check a zip file with your outputs, run `python3 prepare_submit.py path/to/your/output/dir CAMPUSID`, (2) To check your zip file, run `python3 prepare_submit.py path/to/your/submit/zip/file.zip CAMPUSID`

### Grading
* 100: You additionally implement something else on top of the requirements for the score of 100, and achieve significant accuracy improvements. Please write down the things you implemented and experiments you performed in the report. You are also welcome to provide additional materials such as commands to run your code in a script and training logs.
    * perform [continued pre-training](https://arxiv.org/abs/2004.10964) using the MLM objective to do domain adaptation
    * try [alternative fine-tuning algorithms](https://www.aclweb.org/anthology/2020.acl-main.197)
    * add other model components on top of the model
* 95: You implement all the missing pieces and the original ``classifier.py`` with ``--option pretrain`` and ``--option finetune`` code that achieves comparable accuracy to our reference implementation
* 90: You implement all the missing pieces and the original ``classifier.py`` with ``--option pretrain`` and ``--option finetune`` code but accuracy is not comparable to the reference.
* 85: All missing pieces are implemented and pass tests in ``sanity_check.py`` (bert implementation) and ``optimizer_test.py`` (optimizer implementation)
* 80 or below: Some parts of the missing pieces are not implemented.

### Acknowledgement
Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
