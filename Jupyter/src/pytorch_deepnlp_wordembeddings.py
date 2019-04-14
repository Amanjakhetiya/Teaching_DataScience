# -*- coding: utf-8 -*-
# Ref: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)

sentence = "My name is Anthony Gonsalvis"
word_to_ix = {}
for i,word in enumerate(sentence.split()):
    word_to_ix[word] = i

embeds = nn.Embedding(len(word_to_ix.keys()), 5)  # 2 words in vocab, 5 dimensional embeddings
for word in sentence.split():
    lookup_tensor = torch.tensor([word_to_ix[word]], dtype=torch.long)
    hello_embed = embeds(lookup_tensor)
    print(word, " ", hello_embed)