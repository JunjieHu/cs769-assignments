import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

class ToyEncoderDecoderModel(nn.Module):
    def __init__(self, log_prob):
        super().__init__()
        self.batch_size = 3
        self.beam_size = 5
        self.hidden_size = 10
        self.words = ['<s>', '</s>', '<pad>', 'I', 'hate', 'this', 'movie', 'you', 'like', 'that', 'good', 'bad', 'do', 'not']
        self.vocab = {w:i for i, w in enumerate(self.words)}
        self.vocab_size = len(self.vocab)
        self.pad_id = self.vocab['<pad>']
        self.eos_id = self.vocab['</s>']
        self.bos_id = self.vocab['<s>']
        self.log_prob = log_prob

    def encode(self, src_input=None):
        # create simulated encode data
        return torch.rand(self.batch_size, 10, self.hidden_size)

    def decode_one_step(self, t, dec_input, dec_hidden, src_hiddens, top_beamid=None):
        hidden_t = torch.rand(self.batch_size, self.beam_size, self.hidden_size)
        log_prob_t = self.log_prob[:, :, t]
        return log_prob_t, hidden_t
   

def beam_search(model, src_hiddens, beam_size=5, decode_max_length=10, to_word=True):
    batch_size = src_hiddens.size(0)
    vocab_size = len(model.vocab)
    hidden_size = model.hidden_size

    bos_id = model.vocab['<s>']
    eos_id = model.vocab['</s>']
    pad_id = model.vocab['<pad>']

    # Define a pad vector
    pad_vec = torch.FloatTensor(1, vocab_size).fill_(-float('inf'))
    pad_vec[0, pad_id] = 0
    eos_mask = torch.zeros(batch_size, beam_size, dtype=torch.bool)

    # Initialize a decoder input tensor with all <s> and decoder hidden state with all zeros
    dec_input = torch.LongTensor(batch_size, beam_size).fill_(bos_id) # [batch_size, hidden_size]
    dec_hidden = torch.zeros(batch_size, beam_size, hidden_size) # [batch_size, hidden_size]
    
    # Record the cummulated per-word log-likelihood as `top_score`, and back-pointers
    top_score = torch.zeros(batch_size, beam_size)
    top_beamids, top_wordids = [], []
    # Start decoding
    for t in range(decode_max_length):
        # Get the log probability of the next word token, log_prob.shape = [batch_size, beam_size, vocab_size]
        log_prob, dec_hidden = model.decode_one_step(t, dec_input, dec_hidden, src_hiddens)

        # If the partial sequences have generated </s> tokens, replace their next-word log-probability by pad vector.
        if eos_mask is not None and eos_mask.sum() > 0:
            log_prob.masked_scatter_(eos_mask.unsqueeze(2), pad_vec.expand((eos_mask.sum(), vocab_size)))
        
        # Add the next-word log-probability at t-th step to the cummulated top_score tensor.
        score = top_score.unsqueeze(2) + log_prob  # [batch_size, beam_size, vocab_size]

        # Update the top_score, and get the beam/word id at the t-th step.
        top_score, top_beamid, top_wordid = topK(score)  
        top_beamids.append(top_beamid)    # select the beam id
        top_wordids.append(top_wordid)    # select the word id
        
        # Re-arrange the decoder hidden states, which also results in re-arrangement of next-word log-probability
        dec_hidden = select_hiddens_by_beams(dec_hidden, top_beamid)  # [batch, beam_size, hidden_size]
        dec_input = top_wordid  # [batch_size, beam_size]
        model.log_prob = select_hiddens_by_beams(model.log_prob, top_beamid)

        eos_mask = eos_mask.gather(dim=1, index=top_beamid) | top_wordid.eq(eos_id)
        if eos_mask.sum() == batch_size * beam_size:
            break
    
    # Use top_score and back-pointers to extract decoded sequences
    sequences, sort_score = extract_sequences(top_score, top_wordids, top_beamids)
    return sequences, sort_score


def topK(score):
    """
    For every example in a batch, we have generated K candidates (partial sequences) in beam search (K=beam_size).
    For each candidate, we will search for the next token from a vocabulary of V tokens (V=vocab_size). 
    So, we expand the old candidates to get (K * V) new candidates in total, and then we will prune the new
    candidates to only keep the top K candidates in the beam.
    Args: 
        score: torch.FloatTensor, [batch_size, beam_size, vocab_size]. 
               This tensor has the cummulated score of selecting the next token from a vocabuary
               for each old candidate for each example from a batch.
    Return:
        top_score: torch.FloatTensor, [batch_size, beam_size], the scores of the top K new candidates in the beam after pruning 
        top_beamid: torch.LongTensor, [batch_size, beam_size], the beam ids of the top K new candidates
        top_wordid: torch.LongTensor, [batch_size, beam_size], the word ids of the next tokens to construct the top K new candidates
    
    Example:
        Assuming batch_size = 2, beam_size = 3, vocab_size = 4, we have the inputs and outputs as follows:
        score = [[[-0.1, -0.4, -0.5, -0.6], [-0.4, -0.8, -0.4, -0.3], [-0.9, -0.5, -0.2, -0.7]],
                 [[-0.6, -0.3, -0.8, -0.9], [-0.9, -0.5, -0.7, -0.4], [-0.7, -0.8, -0.2, -0.8]]]
        top_score = [[-0.1, -0.2, -0.3], [-0.2, -0.3, -0.4]]
        top_beamid = [[0, 2, 1], [2, 0, 1]]
        top_wordid = [[0, 2, 3], [2, 1, 3]]  
    """
    raise NotImplementedError


def select_hiddens_by_beams(hiddens, beam_id):
    """ Re-arange the hidden state tensors according to the selected beams in the previous step
    Args:
        hiddens: [batch_size, beam_size, hidden_size] 
        beam_id: [batch_size, beam_size] 
    Return:
        new_hiddens: [batch_size, beam_size, hidden_size]
    Example:
        Assuming batch_size = 2, beam_size = 3, hidden_size = 4
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
    """
    raise NotImplementedError


def extract_sequences(top_score, top_wordids, top_beamids):
    """
    Extract generated sequences by traversing (top_wordids and top_beamids) backward.
    For each example in a mini-batch, sort the top K generated sequences in the beam in a descending order.
    Args:
        top_score:  torch.Tensor, [batch_size, beam_size]. Top scores of final generated sequences in the beam. 
        top_wordids: List(torch.LongTensor), each tensor has shape [batch_size, beam_size], the selected word ids of each decoding step 
        top_beamids: List(torch.LongTensor), each tensor has shape [batch_size, beam_size], the selected beam ids of each decoding step 
    Return:
        sequence: torch.Tensor, [batch_size, beam_size, decode_max_length]. word ids of final decoded sequences in the beams.
        sort_score: torch.FloatTensor, [batch_size, beam_size]. the sorted score of the corresponding `sequences` in the beams.
    Example: 
        See inputs and expected outputs in test_extract_sequences().
    """
    raise NotImplementedError

 
