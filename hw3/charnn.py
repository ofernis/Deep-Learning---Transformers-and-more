import re
import random
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    
    chars_bag = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars_bag)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    
    text_clean = text
    n_removed = 0
    
    for c in chars_to_remove:
        text_clean = text_clean.replace(c, '')
        n_removed += text.count(c)
    
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    
    N, D = len(text), len(char_to_idx)

    result = torch.zeros(size=(N, D), dtype=torch.int8)

    for i, c in enumerate(text):
        result[i][char_to_idx[c]] = 1
    
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    
    result = ""
    
    for emb in embedded_text:
        result += idx_to_char[torch.argmax(emb).item()]
    
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    
    embedded_text = chars_to_onehot(text, char_to_idx).to(device)  # N_total, V

    num_of_samples = (embedded_text.shape[0] - 1) // seq_len # N
    
    chars_to_keep = num_of_samples*seq_len
    
    samples_chars = embedded_text[:chars_to_keep, :] # chars_to_keep, V

    labels_chars = embedded_text[1:chars_to_keep+1, :] # chars_to_keep, V
 
    V = embedded_text.shape[1]
    
    samples = samples_chars.reshape(-1, seq_len, V) # N, seq_len, V
    
    labels = labels_chars.reshape(-1, seq_len, V).argmax(dim=-1) # N, seq_len
    
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    
    result = torch.softmax(y / temperature, dim=dim)
    
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======
    
    with torch.no_grad():
        
        hidden_state = None
        
        seq = start_sequence
        
        n_chars -= len(start_sequence)
               
        for _ in range(n_chars):
            embedded_seq = chars_to_onehot(seq, char_to_idx).float().unsqueeze(0).to(device)
            
            layer_output, hidden_state = model(embedded_seq, hidden_state)
            
            logits = hot_softmax(layer_output[0, -1], dim=-1, temperature=T)
            
            sample = torch.multinomial(input=logits, num_samples=1).item()
            
            seq = idx_to_char[sample]
            
            out_text += seq
            
    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        
        num_batches = len(self.dataset) // self.batch_size
        
        idx = torch.arange(num_batches * self.batch_size) \
                   .view(self.batch_size, -1).T \
                   .reshape(1,-1).squeeze().tolist()

        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model for all layers.
        #  To implement the affine transforms you can use either nn.Linear
        #  modules (recommended) or create W and b tensor pairs directly.
        #  Create these modules or tensors and save them per-layer in
        #  the layer_params list.
        #  Important note: You must register the created parameters so
        #  they are returned from our module's parameters() function.
        #  Usually this happens automatically when we assign a
        #  module/tensor as an attribute in our module, but now we need
        #  to do it manually since we're not assigning attributes. So:
        #    - If you use nn.Linear modules, call self.add_module() on them
        #      to register each of their parameters as part of your model.
        #    - If you use tensors directly, wrap them in nn.Parameter() and
        #      then call self.register_parameter() on them. Also make
        #      sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        
        for layer in range(self.n_layers):
            input_dim = self.in_dim if layer == 0 else self.h_dim
            
            Wxz = nn.Linear(in_features=input_dim, out_features=self.h_dim, bias=False)
            self.add_module(name=f"Wxz_{layer}",module=Wxz)

            Whz = nn.Linear(in_features=self.h_dim, out_features=self.h_dim, bias=True)
            self.add_module(name=f"Whz_{layer}",module=Whz)

            Wxr = nn.Linear(in_features=input_dim, out_features=self.h_dim, bias=False)
            self.add_module(name=f"Wxr_{layer}",module=Wxr)

            Whr = nn.Linear(in_features=self.h_dim, out_features=self.h_dim, bias=True)
            self.add_module(name=f"Whr_{layer}",module=Whr)

            Wxg = nn.Linear(in_features=input_dim, out_features=self.h_dim, bias=False)
            self.add_module(name=f"Wxg_{layer}",module=Wxg)

            Whg = nn.Linear(in_features=self.h_dim, out_features=self.h_dim, bias=True)
            self.add_module(name=f"Whg_{layer}",module=Whg)

            dropout = nn.Dropout(p=dropout)
            self.add_module(name=f"Dropout_{layer}",module=dropout)

            self.layer_params.append((Wxz, Whz, Wxr, Whr, Wxg, Whg, dropout))

        # output
        Why = nn.Linear(h_dim, out_dim, bias=True)

        self.add_module(name='Why', module=Why)

        self.layer_params.append(Why)

        # ========================

    
    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO:
        #  Implement the model's forward pass.
        #  You'll need to go layer-by-layer from bottom to top (see diagram).
        #  Tip: You can use torch.stack() to combine multiple tensors into a
        #  single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        m = 0
        
        device = input.device
        
        layer_output = torch.zeros(size=(batch_size, seq_len, self.out_dim), device=device)

        for t in range(seq_len):
            
            x_t = layer_input[:, t, :]
            
            layers_params = self.layer_params[:-1]

            for layer, (params, state) in enumerate(zip(layers_params, layer_states)):
                Wxz_t, Whz_t, Wxr_t, Whr_t, Wxg_t, Whg_t, drop = params
                
                state, x_t = state.to(device), x_t.to(device)
                Wxz_t, Whz_t = Wxz_t.to(device), Whz_t.to(device)
                Wxr_t, Whr_t = Wxr_t.to(device), Whr_t.to(device)
                Wxg_t, Whg_t = Wxg_t.to(device), Whg_t.to(device)
                
                z_t = torch.sigmoid(Wxz_t(x_t) + Whz_t(state))
                r_t = torch.sigmoid(Wxr_t(x_t) + Whr_t(state))
                g_t = torch.tanh(Wxg_t(x_t) + Whg_t(state * r_t))
                
                h_t = z_t * state + (1 - z_t) * g_t
                
                layer_states[layer] = h_t
                
                x_t = drop(h_t)

            wh_y = self.layer_params[-1].to(device=device)
            wh_y = wh_y
            layer_output[:, t, :] = wh_y(x_t)

        hidden_state = torch.stack(layer_states, dim=1)
        
        # ========================
        return layer_output, hidden_state
