import torch
import torch.nn as nn
import torch.nn.functional as F
from beam_search import *
torch.manual_seed(0)

def test_extract_sequences():
    top_score = torch.Tensor([[ 0.00, -0.02, -0.03, -0.03, -0.04],
                              [ 0.00, -0.00, -0.03, -0.03, -0.03],
                              [ 0.00, -0.00, -0.11, -0.11, -0.13]])
    top_wordids = [
        torch.LongTensor([[ 3, 12,  5,  4,  9], [12, 11,  9,  5,  8], [ 3,  8, 11,  4,  5]]), 
        torch.LongTensor([[12,  1,  0, 12, 12], [ 7,  7,  7,  4,  7], [ 4,  0,  2,  7,  4]]), 
        torch.LongTensor([[13,  2, 13, 13, 13], [ 8,  5,  8,  5,  8], [ 9,  9,  2,  2,  9]]), 
        torch.LongTensor([[ 8,  2,  8,  8,  8], [ 5,  5,  5,  5,  5], [11, 11,  1,  1,  9]]), 
        torch.LongTensor([[ 5,  2,  5,  5,  5], [10, 10, 10, 10, 10], [ 6,  6,  2,  2,  0]]), 
        torch.LongTensor([[6, 7, 2, 6, 6], [6, 6, 7, 6, 7], [1, 1, 2, 2, 1]]), 
        torch.LongTensor([[1, 1, 2, 1, 1], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
    ]
    top_beamids = [
        torch.LongTensor([[1, 4, 1, 1, 2], [0, 1, 0, 0, 0], [2, 3, 4, 4, 2]]), 
        torch.LongTensor([[0, 1, 0, 2, 3], [0, 2, 3, 0, 4], [0, 0, 1, 2, 4]]), 
        torch.LongTensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 2], [0, 1, 0, 1, 4]]), 
        torch.LongTensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 0, 1, 0]]), 
        torch.LongTensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 0]]),
        torch.LongTensor([[0, 0, 1, 2, 3], [0, 1, 0, 2, 1], [0, 1, 2, 3, 4]]), 
        torch.LongTensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    ]
    sequence = torch.LongTensor([[[ 3, 12, 13,  8,  5,  6,  1],
                                [ 3, 12, 13,  8,  5,  7,  1],
                                [12,  1,  2,  2,  2,  2,  2],
                                [ 3,  0, 13,  8,  5,  6,  1],
                                [ 5, 12, 13,  8,  5,  6,  1]],

                                [[12,  7,  8,  5, 10,  6,  1],
                                [12,  7,  5,  5, 10,  6,  1],
                                [12,  7,  8,  5, 10,  7,  1],
                                [ 9,  7,  8,  5, 10,  6,  1],
                                [12,  7,  5,  5, 10,  7,  1]],

                                [[ 3,  4,  9, 11,  6,  1,  2],
                                [ 3,  0,  9, 11,  6,  1,  2],
                                [ 3,  4,  9,  1,  2,  2,  2],
                                [ 3,  0,  9,  1,  2,  2,  2],
                                [ 3,  4,  9, 11,  0,  1,  2]]])
    sort_score = torch.FloatTensor([[ 0.00, -0.02, -0.03, -0.03, -0.04],
                                    [ 0.00, -0.00, -0.03, -0.03, -0.03],
                                    [ 0.00, -0.00, -0.11, -0.11, -0.13]])
    pre_sequence, pre_sort_score = extract_sequences(top_score, top_wordids, top_beamids)
    assert torch.allclose(pre_sequence, sequence, rtol=1e-3, atol=1e-04)
    assert torch.allclose(pre_sort_score, sort_score, rtol=1e-3, atol=1e-04)
    print('Pass the test of extract_sequences() function!')
    

def test_topK():
    score = torch.FloatTensor([[[-0.1, -0.4, -0.5, -0.6], [-0.4, -0.8, -0.4, -0.3], [-0.9, -0.5, -0.2, -0.7]],
                [[-0.6, -0.3, -0.8, -0.9], [-0.9, -0.5, -0.7, -0.4], [-0.7, -0.8, -0.2, -0.8]]])
    top_score = torch.FloatTensor([[-0.1, -0.2, -0.3], [-0.2, -0.3, -0.4]])
    top_beamid = torch.LongTensor([[0, 2, 1], [2, 0, 1]])
    top_wordid = torch.LongTensor([[0, 2, 3], [2, 1, 3]])
    pre_score, pre_beamid, pre_wordid = topK(score)
    assert torch.allclose(pre_score, top_score, rtol=1e-3, atol=1e-04)
    assert torch.allclose(pre_beamid, top_beamid, rtol=1e-3, atol=1e-04)
    assert torch.allclose(pre_wordid, top_wordid, rtol=1e-3, atol=1e-04)
    print('Pass the test of topK() function!')


def test_select_hiddens_by_beams():
    hiddens = torch.FloatTensor([[[0.6566, 0.2719, 0.7182, 0.1083],
                                [0.1627, 0.4812, 0.1167, 0.4318],
                                [0.1817, 0.2578, 0.5769, 0.2610]],

                                [[0.9372, 0.4993, 0.5471, 0.9169],
                                [0.8798, 0.6168, 0.6097, 0.8790],
                                [0.6642, 0.4301, 0.5542, 0.3670]]])
    beam_id = torch.LongTensor([[1, 1, 0], [2, 0, 0]])
    new_hiddens = torch.FloatTensor([[[0.1627, 0.4812, 0.1167, 0.4318],
                                    [0.1627, 0.4812, 0.1167, 0.4318],
                                    [0.6566, 0.2719, 0.7182, 0.1083]],

                                    [[0.6642, 0.4301, 0.5542, 0.3670],
                                    [0.9372, 0.4993, 0.5471, 0.9169],
                                    [0.9372, 0.4993, 0.5471, 0.9169]]])
    pre_hiddens = select_hiddens_by_beams(hiddens, beam_id)
    assert torch.allclose(pre_hiddens, new_hiddens, rtol=1e-3, atol=1e-04)
    print('Pass the test of select_hiddens_by_beams() function!')


def test_beam_search():
    data = torch.load('sanity_check.pt')
    model = ToyEncoderDecoderModel(data['log_prob'])
    src_hiddens = model.encode()
    sequence, sort_score = beam_search(model, src_hiddens)
    assert torch.allclose(sequence, data['sequence'], rtol=1e-3, atol=1e-04)
    assert torch.allclose(sort_score, data['sort_score'], rtol=1e-3, atol=1e-04)
    print('Pass the test of beam_search() function!')

    sequence = sequence.tolist()
    sequence = [[[model.words[wid] for wid in seq if wid != model.vocab['<pad>']] for seq in beam] for beam in sequence]
    sort_score = sort_score.tolist()
    # Output sequence
    for i, (seq, score) in enumerate(zip(sequence, sort_score)):
        print(f'Sentence-{i}:')
        for j, s in enumerate(seq):
            print(f'  * Beam-{j}, score={score[j]:.4f}: {" ".join(s)}')


test_topK()
test_select_hiddens_by_beams()
test_extract_sequences()
test_beam_search()
