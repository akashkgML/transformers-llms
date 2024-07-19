#!/usr/bin/python3

"""
BERT model from scratch.
"""
# References
# ----------
# 1. https://github.com/jadore801120/attention-is-all-you-need-pytorch
# 2. https://github.com/JayParks/transformer
# 3. https://github.com/dhlee347/pytorchic-bert
# 4. https://arxiv.org/abs/1706.03762 - Attention is all you need paper

import re
from typing import List, Dict, Tuple
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def get_clean_text(text: str) -> str:
    """
    Function to remove punctuation and special characters
    from given text data. Its aims to clean the text data
    and retain only alphanumeric characters and spaces
    between words.

    :param text: textual corpus
    :return: clean textual Corpus
    """
    clean_text = re.sub(pattern='[^a-zA-Z0-9\\s]', repl='', string=text.lower())
    return clean_text


def sentences_from_text(text: str) -> List[str]:
    """
    Function to split text into list of sentences.

    :param text: clean text corpus
    :return: list of sentences
    """
    sentences = text.split('\n')
    return sentences


def get_word_list(sentences: List[str]) -> List[str]:
    """
    Function to create list of words from list of
    sentences.

    :param sentences: list of sentences
    :return: list of words
    """
    words = list(set(" ".join(sentences).split()))
    return words


def create_word_to_id_dict(words, special_tokens):
    # type: (List[str], Dict[str, int]) -> Dict[str, int]
    """
    Function to create dictionary from list of words
    having word as a key and values as an index in the
    vocabulary.

    :param words: list of words from the corpus
    :param special_tokens: dict of special tokens
    :return: word-indices dictionary
    """
    # get number of tokens in special dict
    spl_tok_size = len(special_tokens)
    word_dict = {}
    word_dict.update(special_tokens)
    for i, w in enumerate(words):
        word_dict[w] = i + spl_tok_size
    return word_dict


def create_id_to_word_dict(word_to_id):
    # type: (Dict[str, int]) -> Dict[int, str]
    """
    Function to create dictionary from word to id dict
    where id is key and word is value.

    :param word_to_id: dict containing words as keys and values as index
    :return: dict with index as keys and words as values
    """
    num_dict = {v: k for k, v in word_to_id.items()}
    return num_dict


def get_sentences_as_tokens_ids(sentences, word_dict):
    # type: (List[str], Dict[str, int]) -> List[List[int]]
    """
    Function to represent sentences as a sequence of token ids.

    :param sentences: list of sentences
    :param word_dict: dict having keys as words and values as indices
    :return: list of sentences represented as a sequence of token ids
    """
    # create an empty token list
    tokens_list = list()
    # iterate through each sentence
    for sentence in sentences:
        tokens = [word_dict[token] for token in sentence.split()]
        tokens_list.append(tokens)

    return tokens_list


def get_sentence_pairs(tokens_list, batch_size):
    # type: (List[List[int]], int) -> List[List[List[int] | bool]]
    """
    Function to generate connected sentences pairs as well as disconnected
    or random sentence pairs along with a boolean label. The label is true
    if sentences are connected else it is false. This dataset is used for
    Next Sentence Prediction Task. The dataset will contain 50% as connected
    sentences and other 50% as random sentences.

    :param tokens_list: list of sentences represented as token ids
    :param batch_size: size of the batch for pre-training
    :return: list of pairs of sentences(as token ids) with a boolean label
    """
    pairs_with_label = []
    num_sentences = len(tokens_list)
    # we need to create connected as well as random sentence pairs
    num_connected_pairs = num_random_pairs = 0

    # we need 50% as the connected pairs and rest 50% as random pairs
    required_negative_samples = batch_size // 2
    required_positive_samples = batch_size - required_negative_samples

    while num_connected_pairs != required_positive_samples or num_random_pairs != required_negative_samples:
        tokens_idx_a, tokens_idx_b = random.randrange(num_sentences), random.randrange(num_sentences)
        # check if connected sentences pair
        if tokens_idx_a + 1 == tokens_idx_b and num_connected_pairs < required_positive_samples:
            # isNext is True
            pairs_with_label.append([tokens_list[tokens_idx_a], tokens_list[tokens_idx_b], True])
            num_connected_pairs += 1
        # sentence pairs are random or disconnected
        elif num_random_pairs < required_negative_samples:
            # NotNext is False
            pairs_with_label.append([tokens_list[tokens_idx_a], tokens_list[tokens_idx_b], False])
            num_random_pairs += 1
    return pairs_with_label


def create_input_ids(tokens_a, tokens_b, word_to_id):
    # type: (List[int], List[int], Dict[str, int]) -> List[int]
    """
    Function to create input sequence by adding special tokens
    as per the format of BERT.

    :param tokens_a: token ids for sentence 'a'
    :param tokens_b: token ids for sentence 'b'
    :param word_to_id: dict having keys as words and values as indices
    :return: single formatted sequence of input
    """
    input_ids = [word_to_id['[CLS]']] + tokens_a + \
                [word_to_id['[SEP]']] + tokens_b + [word_to_id['[SEP]']]

    return input_ids


def create_segment_ids(tokens_a, tokens_b):
    # type: (List[int], List[int]) -> List[int]
    """
    Function to create segment ids denoting whether tokens
    belong to sentence 'a' or sentence 'b'.

    :param tokens_a: token ids for sentence 'a'
    :param tokens_b: token ids for sentence 'b'
    :return: sequence of segment ids
    """
    # segment id is 0 for tokens belonging to sentence 'a' and 1 for sentence 'b'
    segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

    return segment_ids


def mask_sequence(input_ids, max_pred, word_to_id_dict, id_to_word_dict, mask_lm_prob=0.15):
    # type: (List[int], int, Dict[str, int], Dict[int, str], float) -> Tuple[List[int], List[int], List[int], int]
    """
    Function to mask sentences. 15% of the tokens are randomly replaced chosen.
    Out of which 80% of the times the ask token is replaced by the mask token,
    10% of the time it is replaced by random token and rest 10% of the time it
    is unchanged.

    :param input_ids: input sequence id
    :param max_pred: maximum number of tokens to predict
    :param word_to_id_dict: dict containing keys as words and values as ids
    :param id_to_word_dict: dict containing keys as ids and values as words
    :param mask_lm_prob: percentage of tokens to be masked (15% default)
    :return: tuple of input ids, masked tokens, masked positions and final number of
     tokens to predict
    """
    # vocab size
    vocab_size = len(word_to_id_dict)
    # mask 15% of the tokens are to be masked
    num_pred = min(max_pred, max(1, int(round(len(input_ids)) * mask_lm_prob)))
    cand_mask_pos = [i for i, token in enumerate(input_ids)
                     if token != word_to_id_dict['[CLS]']
                     and token != word_to_id_dict['[SEP]']]
    # shuffle the candidate masked position
    random.shuffle(cand_mask_pos)
    masked_tokens, masked_pos = [], []
    for pos in cand_mask_pos[:num_pred]:
        masked_pos.append(pos)
        masked_tokens.append(input_ids[pos])
        # 80% of the time, we replace the token with [MASK]
        if random.random() < 0.8:
            input_ids[pos] = word_to_id_dict['[MASK]']
        # 10% of the time replace it with random word
        elif random.random() > 0.5:
            # choose random index in vocabulary
            random_index = random.randint(0, vocab_size - 1)
            input_ids[pos] = word_to_id_dict[id_to_word_dict[random_index]]
        # 10% of the time we leave the token unchanged and hence nothing to do
    return input_ids, masked_tokens, masked_pos, num_pred


def add_paddings(input_ids, segment_ids, masked_tokens, masked_positions, max_len, max_pred, num_pred):
    # type: (List[int], List[int], List[int], List[int], int, int, int) -> Tuple[List[int], List[int], List[int], List[int]]
    """
    Functon to add padding tokens to make sequences of equal lengths

    :param input_ids: input id sequence
    :param segment_ids: segment id sequence of 0 and 1
    :param masked_tokens: masked tokens
    :param masked_positions: positions of masked tokens
    :param max_pred: maximum number of tokens to predict
    :param num_pred: final number of tokens to predict
    :return: tuple of input ids, segment ids, masked tokens and their positions
    """
    num_pad = max_len - len(input_ids)
    input_ids.extend([0] * num_pad)
    segment_ids.extend([0] * num_pad)

    # Zero paddings (100% - 15%)
    if max_pred > num_pred:
        num_pad = max_pred - num_pred
        masked_tokens.extend([0] * num_pad)
        masked_positions.extend([0] * num_pad)

    return input_ids, segment_ids, masked_tokens, masked_positions


def create_batch(tokens_list, batch_size, word_to_id, id_to_word, max_pred):
    # type: (List[List[int]], int, Dict[str, int], dict[int, str], int) -> [List[List[List[int]| bool]]]
    """
    Function to generate batches of pre-training dataset.

    :param tokens_list: list of token ids representing sentences
    :param batch_size: size of batch
    :param word_to_id: dict having keys as words and values as indices
    :param id_to_word: dict having keys as indices and values as words
    :param max_pred: maximum number of tokens to predict
    :return: pre-training batch
    """
    # create an empty list for batch
    batch = []
    # ---Next Sentence Prediction Tasks---
    # Get sentence pairs with labels isNext and NotNext
    sentence_pairs_with_labels = get_sentence_pairs(tokens_list, batch_size)
    # iterate through each sentence pair with label attached
    for pair_label in sentence_pairs_with_labels:
        tokens_a, tokens_b, label = pair_label[0], pair_label[1], pair_label[2]
        # Add CLS and SEP special tokens to form input ids
        input_ids = create_input_ids(tokens_a, tokens_b, word_to_id)
        # create segment ids
        segment_ids = create_segment_ids(tokens_a, tokens_b)
        # ---Masked Language Modeling Tasks---
        # Mask the sequence in consideration
        (input_ids, masked_tokens, masked_pos, num_pred) = mask_sequence(
            input_ids, max_pred, word_to_id, id_to_word)
        # Add paddings to make sequences of equal length
        (input_ids, segment_ids, masked_tokens, masked_pos) = add_paddings(
            input_ids, segment_ids, masked_tokens, masked_pos,
            max_len,max_pred, num_pred)
        # append the data instance to batch
        batch.append([input_ids, segment_ids, masked_tokens, masked_pos, label])

    return batch


def get_attention_pad_mask(seq_query, seq_key):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Function to extract padded attention mask.

    :param seq_query: sequence of query
    :param seq_key: sequence of key
    :return: padded attention mask
    """
    # size of the sequence query tensor
    batch_size, len_query = seq_query.size()
    # size of the sequence key tensor
    batch_size, len_key = seq_key.size()
    # padded token equals 0
    pad_attn_mask = seq_key.data.eq(0).unsqueeze(1)

    return pad_attn_mask.expand(batch_size, len_query, len_key)


def gelu(x):
    # type: (torch.Tensor) -> torch.Tensor
    """
    The Gaussian error linear unit (GELU) activation
    operation weights the input by its probability
    under a Gaussian distribution.

    :param x: input tensor (dim = batch_size x len_seq x d_model)
    :return: non-linear output (of same dimension as x)
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(nn.Module):
    """
    A class to represent embedding layers of different types and normalization layer.
    """
    def __init__(self, vocab_size, d_model, max_len, num_segments):
        # type: (int, int, int, int) -> None
        super(Embedding, self).__init__()
        # attribute for token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # attribute for position embedding
        self.position_embedding = nn.Embedding(max_len, d_model)
        # attribute for segment embedding
        self.segment_embedding = nn.Embedding(num_segments, d_model)
        # attribute for layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward functon defining the embedding network structure and how
        the network is going to be run.

        :param x: input tensor
        :param seg: segment tensor
        :return: normalized input embedding
        """
        sequence_len = x.size(1)
        position = torch.arange(sequence_len, dtype=torch.long)
        position = position.unsqueeze(0).expand_as(x)
        # input embedding is the sum of token emb + position emb + segment emb
        input_embedding = self.token_embedding(x) + self.position_embedding(position) + self.segment_embedding(seg)
        # normalize the input embedding
        return self.norm(input_embedding)


class ScaledDotProductAttention(nn.Module):
    """
    A class to help compute Scaled Dot Product attention.
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Query, Key, Value, attention_mask, dim_key):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int) -> Tuple[torch.Tensor, torch.Tensor]
        """
        Forward functon defining the ScaledDotProductAttention network
        structure and how the network is going to be run.

        :param Query: query tensor
        :param Key: key tensor
        :param Value: Value Tensor
        :param attention_mask: attention mask tensor
        :param dim_key: dimension of key
        :return: context, attention
        """
        # multiply Query with Key transpose to get scores i.e. scores = Q.K_T
        scores = torch.matmul(Query, Key.transpose(-1, -2)) / np.sqrt(dim_key)
        # Fills element of self tensor with value -∞ where mask is one
        scores.masked_fill_(attention_mask, -1e9)
        # Apply softmax to scaled dot product i.e. Softmax of Q.K_t/√(dim_key)
        attention = nn.Softmax(dim=-1)(scores)
        # compute the context i.e. Softmax(Q.K_t/√(dim_key)) * V
        context = torch.matmul(attention, Value)

        return context, attention


class MultiHeadAttention(nn.Module):
    """
    A class to represent Multi Head Attention Layer. Also stores
    weight tensors for query, key and values.
    """
    def __init__(self, dim_model, dim_key, dim_value, num_heads):
        # type: (int, int, int, int) -> None
        # Query Weight Tensor
        super(MultiHeadAttention, self).__init__()
        self.dim_model = dim_model
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.num_heads = num_heads
        # create and initialize the different weight layers
        self._init_layers()

    def _init_layers(self):
        self.W_Q = nn.Linear(self.dim_model, self.dim_key * self.num_heads)
        # Key Weight Tensor
        self.W_K = nn.Linear(self.dim_model, self.dim_key * self.num_heads)
        # Value Weight Tensor
        self.W_V = nn.Linear(self.dim_model, self.dim_value * self.num_heads)

    def forward(self, Query, Key, Value, attention_mask):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        """
        Forward functon defining the MultiHeadAttention network
        structure and how the network is going to be run.

        :param Query: Query Tensor
        :param Key: Key Tensor
        :param Value: Value Tensor
        :param attention_mask:
        :return: normalized output, attention
        """
        residual, batch_size = Query, Query.size(0)
        # Compute query: [batch_size x num_heads x len_query x dim_key]
        query = self.W_Q(Query).view(batch_size, -1, self.num_heads, self.dim_key).transpose(1, 2)
        # Compute key: [batch_size x num_heads x len_key x dim_key]
        key = self.W_K(Key).view(batch_size, -1, self.num_heads, self.dim_key).transpose(1, 2)
        # Compute value: [batch_size x num_heads x len_key x dim_value]
        value = self.W_V(Value).view(batch_size, -1, self.num_heads, self.dim_value).transpose(1, 2)
        # compute attention mask: [batch_size x num_heads x len_query x len_key]
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        # Note: len_key=len_query
        # compute context: [batch_size x num_heads x len_query x dim_value] and
        # compute attention: [batch_size x num_heads x len_query x len_key]
        context, attention = ScaledDotProductAttention()(query, key, value, attention_mask, self.dim_key)
        # context: [batch_size x len_query x num_heads x dim_value]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_value)
        # output: [batch_size x len_query x dim_model]
        output = nn.Linear(self.num_heads * self.dim_value, self.dim_model)(context)

        return nn.LayerNorm(self.dim_model)(output + residual), attention


class PositionWiseFeedForwardNet(nn.Module):
    """
    The Position-wise Feed-Forward Network is an expansion-and-contraction network
    that transforms each sequence using the same dense layers.
    """
    def __init__(self, dim_model, dim_ff):
        # type: (int, int) -> None
        super(PositionWiseFeedForwardNet, self).__init__()
        # fully connected layer 1
        self.fc1 = nn.Linear(dim_model, dim_ff)
        # fully connected layer
        self.fc2 = nn.Linear(dim_ff, dim_model)

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward functon defining the PositionWiseFeedForwardNet network
        structure and how the network is going to be run.

        :param x: tensor from MHA layer
        :return: tensor
        """
        # [batch_size, len_seq, dim_model] --fc1--> [batch_size, len_seq, dim_ff]
        # [batch_size, len_seq, dim_ff] --fc2(gleu(.))--> [batch_size, len_seq, dim_model]
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """
    An encoder block is where all the beautiful stuff happens. The encoder in
    the transformer consists of multiple encoder blocks. An input sentence goes
    through the encoder blocks, and the output of the last encoder block becomes
    the input features to the decoder.
    """
    def __init__(self, dim_model, dim_key, dim_value, num_heads, dim_ff):
        # type: (int, int, int, int, int) -> None
        super(EncoderLayer, self).__init__()
        # attribute for encoder multi-head attention layer
        self.encoder_mha = MultiHeadAttention(dim_model, dim_key, dim_value, num_heads)
        # attribute for position wise feed forward neural net layer
        self.position_ffn = PositionWiseFeedForwardNet(dim_model, dim_ff)

    def forward(self, encoder_inputs, encoder_mha_mask):
        # type: (torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        """
        Forward functon defining the Encoder's network structure
        and how the network is going to be run.

        :param encoder_inputs: input tensor to the encoder.
        :param encoder_mha_mask: encoder mask multi-head attention tensor.
        :return: tuple of encoder output tensor and attention tensor
        """
        # three copies of encoder inputs as Query, Key and Value
        encoder_outputs, attention = self.encoder_mha(
            encoder_inputs, encoder_inputs, encoder_inputs, encoder_mha_mask)
        # encoder_outputs: [batch_size x len_query x dim_model]
        encoder_outputs = self.position_ffn(encoder_outputs)

        return encoder_outputs, attention


# ---Bert Model--- #
class BERT(nn.Module):
    """
    The main class representing the BERT model. It consists of all the components,
    mainly the encoder and the decoder block.
    """
    def __init__(self, vocab_size, dim_model, dim_key, dim_value, num_heads, dim_ff, max_len, num_segments, num_layers):
        # type: (int, int, int, int, int, int, int, int, int) -> None
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size, dim_model, max_len, num_segments)
        self.layers = nn.ModuleList([EncoderLayer(dim_model, dim_key,
                                                  dim_value, num_heads,
                                                  dim_ff) for _ in range(num_layers)])
        self.fc = nn.Linear(dim_model, dim_model)
        self.activation1 = nn.Tanh()
        self.linear = nn.Linear(dim_model, dim_model)
        self.activation2 = gelu
        self.norm = nn.LayerNorm(dim_model)
        self.classifier = nn.Linear(dim_model, 2)
        # decoder is shared with embedding layer
        embedding_weight = self.embedding.token_embedding.weight
        num_vocab, num_dim = embedding_weight.size()
        # define the decoder structure
        self.decoder = nn.Linear(num_dim, num_vocab, bias=False)
        self.decoder.weight = embedding_weight
        self.decoder_bias = nn.Parameter(torch.zeros(num_vocab))

    def forward(self, input_ids, segment_ids, masked_positions):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        """
        Forward functon defining the BERT's Network Structure
        and how the network is going to be run.

        :param input_ids: tensor of input id sequence
        :param segment_ids: tensor of segment id sequence
        :param masked_positions: tensor for masked positions
        :return: tuple of logit language model and logit classifier model
        """
        output = self.embedding(input_ids, segment_ids)
        encoder_self_attention_mask = get_attention_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            # output: [batch_sze, len_query, dim_model], self_Attention: [batch_size, num_heads, dim_model, dim_model]
            output, encoder_self_attention = layer(output, encoder_self_attention_mask)
        # output will be diced first token [CLS], pool: [batch_size, dim_model]
        h_pooled = self.activation1(self.fc(output[:, 0]))
        # since binary classifier, hence [batch_size, 2]
        logits_clsf = self.classifier(h_pooled)

        # masked positions : [batch_size, max_pred, dim_model]
        masked_positions = masked_positions[:, :, None].expand(-1, -1, output.size(-1))
        # get masked positions from final output of transformer.
        # masking position: [batch_size, max_pred, dim_model]
        h_masked = torch.gather(output, 1, masked_positions)
        h_masked = self.norm(self.activation2(self.linear(h_masked)))
        # [batch_size, max_pred, num_vocab]
        logits_lm = self.decoder(h_masked) + self.decoder_bias

        return logits_lm, logits_clsf


# main driver code
if __name__ == "__main__":

    # the pre-training raw corpus
    corpus = (
        'Hello, How are you? I am Alice.\n'
        'Hello Alice, My name is Akash. Nice to meet you.\n'
        'Nice meet you too. How are you today?\n'
        'Great. My football team won the tournament.\n'
        'Oh Congratulations, Akash.\n'
        'Thank you Alice!'
    )

    # maximum length of sentence (i.e. num of words/tokens)
    max_len = 30

    # batch size for training
    batch_size = 6

    # maximum number of tokens to predict
    max_pred = 5

    # Number of encoders in Encoder Layer
    N = 6

    # Number of heads in multi-head attention
    H = 12

    # Size of the Embedding
    dim_model = 768

    # dimension of Key K (= dimension of Query Q)
    dim_key = int(dim_model / H)

    # dimension of Value V
    dim_value = dim_key

    # dimension of feedforward network
    dim_ff = 4 * dim_model

    # number of segments
    num_segments = 2

    # define special tokens dictionary
    special_tokens_dict = {
        '[PAD]': 0,
        '[CLS]': 1,
        '[SEP]': 2,
        '[MASK]': 3
    }

    print(f"Given Corpus:\n{corpus}\n")

    # get clean corpus
    clean_corpus = get_clean_text(corpus)
    print(f"Clean Corpus:\n{clean_corpus}\n")

    # get list of sentences from the cleaned corpus
    sentences = sentences_from_text(clean_corpus)
    print(f"List of sentences in the corpus:\n{sentences}\n")

    # get the word list from the sentences
    word_list = get_word_list(sentences)
    print(f'Total number of unique words in the corpus: {len(word_list)}')
    print(f"{word_list}\n")

    # create the word to index(ids) dictionary
    word_to_id = create_word_to_id_dict(word_list, special_tokens_dict)
    print(f"word to index(ids) dictionary:\n{word_to_id}\n")

    # create the index(ids) to word dictionary
    id_to_word = create_id_to_word_dict(word_to_id)
    print(f"index(ids) to word dictionary:\n{id_to_word}\n")

    # get sentences as list of tokens ids
    sentence_token_ids = get_sentences_as_tokens_ids(sentences, word_to_id)
    print(f"list of token ids representing sentences:\n{sentence_token_ids}\n")

    # get pairs of sentences with label(true/false) for Next Sentence Prediction Task
    sentence_pairs = get_sentence_pairs(sentence_token_ids, 6)
    print(f"list of pairs of sentences - 50% connected(True), 50% random(False):\n{sentence_pairs}\n")

    # create the pre-training batch
    batch = create_batch(sentence_token_ids, batch_size, word_to_id, id_to_word, max_pred)
    print(f"Pre-training batch:\n{batch}\n")

    # size of vocabulary
    vocab_size = len(word_to_id)

    # create the BERT model
    model = BERT(vocab_size=vocab_size, dim_model=dim_model, dim_value=dim_value, dim_key=dim_key,
                 num_heads=H, dim_ff=dim_ff, max_len=max_len, num_segments=num_segments, num_layers=N)

    # loss criterion as cross entropy loss
    loss_criterion = nn.CrossEntropyLoss()

    # optimizer as Adam Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # extract the input ids, segment ids, masked tokens and positions, isNext as Long Tensors
    input_ids, segment_ids, masked_tokens, masked_positions, isNext = map(torch.LongTensor, zip(*batch))

    # -----Training Phase-----
    print("Training-phase starts..")
    for epoch in range(100):
        # reset the gradient
        optimizer.zero_grad()
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_positions)
        # loss for masked language modelling (i.e. MLM)
        loss_lm = loss_criterion(logits_lm.transpose(1, 2), masked_tokens)
        # take the mean of the loss for MLM
        loss_lm = (loss_lm.float()).mean()
        # loss for binary sentence classification (i.e. NSP)
        loss_clsf = loss_criterion(logits_clsf, isNext)
        # Net Loss is the sum of MLM loss and NSP Loss
        loss = loss_lm + loss_clsf
        # display the cost per epoch
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {(epoch + 1)}, cost = {loss:.03f}")
        # run backpropagation
        loss.backward()
        # perform a single optimization step
        optimizer.step()

    # -----Predict Masked Tokens and IsNext-----
    # extract the test sample input ids, segment ids, masked tokens and positions, isNext as Long Tensors
    test_input_id, test_segment_id, test_masked_tokens, test_masked_positions, test_isNext = map(torch.LongTensor, zip(batch[0]))
    print(f"\nsample test input:\n{test_input_id}\n")

    print("sample with mask tokens:")
    print([id_to_word[word.item()] for word in test_input_id[0] if id_to_word[word.item()] != '[PAD]'])

    # Get Masked Language Model(MLM) and Next Sentence Prediction Model(NSP)
    logits_lm, logits_clsf = model(test_input_id, test_segment_id, test_masked_positions)
    logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
    print('list of masked token indices : ', [pos.item() for pos in test_masked_tokens[0] if pos.item() != 0])
    print('list of predicted masked tokens : ', [pos for pos in logits_lm if pos != 0])

    logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]

    print('isNext : ', True if test_isNext else False)
    print('predicted isNext : ', True if logits_clsf else False)
