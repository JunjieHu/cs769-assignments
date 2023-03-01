import torch
from types import SimpleNamespace
from bert import BertModel, BertSelfAttention, BertLayer 
from utils import *
sanity_data = torch.load("./sanity_check.data")


# Unit Test for BertModel
# text_batch = ["hello world", "hello neural network for NLP"]
# tokenizer here
sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                         [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])
att_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],[1, 1, 1, 1, 1, 1, 1, 1]])

# load our model
bert = BertModel.from_pretrained('bert-base-uncased')
bertlayer = bert.bert_layers[0]
bert_attn = bertlayer.self_attention
bert.eval()
bertlayer.eval()
bert_attn.eval()

# 1. BertModel.embed function
embed_outputs = bert.embed(input_ids=sent_ids)
assert torch.allclose(embed_outputs, sanity_data['embed_outputs'], rtol=1e-3, atol=1e-04)
print("Your BertModel.embed() implementation is correct!")

# 2. BertSoftAttention
hidden_states = embed_outputs
attention_mask: torch.Tensor = get_extended_attention_mask(att_mask, bert.dtype)
attn_outputs = bert_attn(hidden_states, attention_mask)
assert torch.allclose(attn_outputs, sanity_data['attn_outputs'], rtol=1e-3, atol=1e-04)
print("Your BertSelfAttention implementation is correct!")

# 2. BertLayer
layer_outputs = bertlayer(hidden_states, attention_mask)
assert torch.allclose(layer_outputs, sanity_data['layer_outputs'], rtol=1e-3, atol=1e-04)
print("Your BertLayer implementation is correct!")

# 4. BertModel
outputs = bert(sent_ids, att_mask)
for k in ['last_hidden_state', 'pooler_output']:
    assert torch.allclose(outputs[k], sanity_data[k], rtol=1e-3, atol=1e-4)
print("Your BERT implementation is correct!")
