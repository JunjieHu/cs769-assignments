from collections import defaultdict
import os
import time
import random
import torch
import torch.nn as nn
import model as mn
import numpy as np
import argparse
from vocab import Vocab


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/sst-train.txt")
    parser.add_argument("--dev", type=str, default="data/sst-dev.txt")
    parser.add_argument("--test", type=str, default="data/sst-test.txt")
    parser.add_argument("--emb_file", type=str, default=None)
    parser.add_argument("--emb_size", type=int, default=300)
    parser.add_argument("--hid_size", type=int, default=300)
    parser.add_argument("--hid_layer", type=int, default=3)
    parser.add_argument("--word_drop", type=float, default=0.3)
    parser.add_argument("--emb_drop", type=float, default=0.333)
    parser.add_argument("--hid_drop", type=float, default=0.333)
    parser.add_argument("--pooling_method", type=str, default="avg", choices=["sum", "avg", "max"])
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--max_train_epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lrate", type=float, default=0.005)
    parser.add_argument("--lrate_decay", type=float, default=0)  # 0 means no decay!
    parser.add_argument("--mrate", type=float, default=0.85)
    parser.add_argument("--log_niter", type=int, default=100)
    parser.add_argument("--eval_niter", type=int, default=500)
    parser.add_argument("--model", type=str, default="model.pt")  # save/load model name
    parser.add_argument("--dev_output", type=str, default="output.dev.txt")  # output for dev
    parser.add_argument("--test_output", type=str, default="output.test.txt")  # output for dev
    args = parser.parse_args()
    print(f"RUN: {vars(args)}")
    return args

def read_dataset(filename):
    dataset = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            dataset.append((words.split(' '), tag))
    return dataset

def convert_text_to_ids(dataset, word_vocab, tag_vocab):
    data = []
    for words, tag in dataset:
        word_ids = [word_vocab[w] for w in words]
        data.append((word_ids, tag_vocab[tag]))
    return data

def data_iter(data, batch_size, shuffle=True):
    """
    Randomly shuffle training data, and partition into batches.
    Each mini-batch may contain sentences with different lengths.
    """
    if shuffle:
        # Shuffle training data.
        np.random.shuffle(data)

    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
        tags = [data[i * batch_size + b][1] for b in range(cur_batch_size)]
        yield sents, tags

def pad_sentences(sents, pad_id):
    """
    Adding pad_id to sentences in a mini-batch to ensure that 
    all augmented sentences in a mini-batch have the same word length.
    Args:
        sents: list(list(int)), a list of a list of word ids
        pad_id: the word id of the "<pad>" token
    Return:
        aug_sents: list(list(int)), |s_1| == |s_i|, for s_i in sents
    """
    raise NotImplementedError()

def compute_grad_norm(model, norm_type=2):
    """
    Compute the gradients' L2 norm
    """
    total_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        p_norm = p.grad.norm(norm_type) ** (norm_type)
        total_norm += p_norm
    return total_norm ** (1. / norm_type)


def compute_param_norm(model, norm_type=2):
    """
    Compute the model's parameters' L2 norm
    """
    total_norm = 0.0
    for p in model.parameters():
        p_norm = p.norm(norm_type) ** (norm_type)
        total_norm += p_norm
    return total_norm ** (1. / norm_type)

def evaluate(dataset, model, device, tag_vocab=None, filename=None):
    """
    Evaluate test/dev set
    """
    model.eval()
    predicts = []
    acc = 0
    for words, tag in dataset:
        X = torch.LongTensor([words]).to(device)
        scores = model(X)
        y_pred = scores.argmax(1)[0].item()
        predicts.append(y_pred)
        acc += int(y_pred == tag)
    print(f'  -Accuracy: {acc/len(predicts):.4f} ({acc}/{len(predicts)})')
    if filename:
        with open(filename, 'w') as f:
            for y_pred in predicts:
                # convert tag_id to its original label
                tag = tag_vocab.id2word[y_pred]
                f.write(f'{tag}\n')
        print(f'  -Save predictions to {filename}')
    model.train()
    return acc/len(predicts)

def main():
    args = get_args()
    _seed = os.environ.get("MINNN_SEED", 12341)
    random.seed(_seed)
    np.random.seed(_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read datasets
    train_text = read_dataset(args.train)
    dev_text = read_dataset(args.dev)
    test_text = read_dataset(args.test)
    # Build vocabularies for words and tags from training data
    word_vocab = Vocab(pad=True, unk=True)
    word_vocab.build(list(zip(*train_text))[0])
    tag_vocab = Vocab()
    tag_vocab.build(list(zip(*train_text))[1])
    # Convert word string to word ids
    train_data = convert_text_to_ids(train_text, word_vocab, tag_vocab)
    dev_data = convert_text_to_ids(dev_text, word_vocab, tag_vocab)
    test_data = convert_text_to_ids(test_text, word_vocab, tag_vocab)

    # Create a model
    nwords = len(word_vocab)
    ntags = len(tag_vocab)
    print('nwords', nwords, 'ntags', ntags)
    model = mn.DanModel(args, word_vocab, len(tag_vocab)).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lrate, lr_decay=args.lrate_decay)

    # Training
    start_time = time.time()
    train_iter = 0
    train_loss = train_example = train_correct = 0
    best_records = (0, 0)  # [best_iter, best_accuracy]
    for epoch in range(args.max_train_epoch):
        for batch in data_iter(train_data, batch_size=args.batch_size, shuffle=True):
            train_iter += 1

            X = pad_sentences(batch[0], word_vocab['<pad>'])
            X = torch.LongTensor(X).to(device)
            Y = torch.LongTensor(batch[1]).to(device)
            # Forward pass: compute the unnormalized scores for P(Y|X)
            scores = model(X)
            loss = loss_func(scores, Y)
            # Backpropagation: compute gradients for all parameters
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # Update model's parameters with gradients
            optimizer.step()

            train_loss += loss.item() * len(batch[0])
            train_example += len(batch[0])
            Y_pred = scores.argmax(1)
            train_correct += (Y_pred == Y).sum().item()

            if train_iter % args.log_niter == 0:
                gnorm = compute_grad_norm(model)
                pnorm = compute_param_norm(model)
                print(f'Epoch {epoch}, iter {train_iter}, train set: '\
                    f'loss={train_loss/train_example:.4f}, '\
                    f'accuracy={train_correct/train_example:.2f} ({train_correct}/{train_example}), '\
                    f'gradient_norm={gnorm:.2f}, params_norm={pnorm:.2f}, '\
                    f'time={time.time()-start_time:.2f}s')
                train_loss = train_example = train_correct = 0

            if train_iter % args.eval_niter == 0:
                print(f'Evaluate dev data:')
                dev_accuracy = evaluate(dev_data, model, device) 
                if dev_accuracy > best_records[1]:
                    print(f'  -Update best model at {train_iter}, dev accuracy={dev_accuracy:.4f}')
                    best_records = (train_iter, dev_accuracy)
                    model.save(args.model)

    # Load the best model
    model.load(args.model)
    evaluate(test_data, model, device, tag_vocab, filename=args.test_output)
    evaluate(dev_data, model, device, tag_vocab, filename=args.dev_output)


if __name__ == '__main__':
    main()
