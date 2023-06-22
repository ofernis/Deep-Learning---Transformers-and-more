r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 512
    hypers['seq_len'] = 64
    hypers['h_dim'] = 1024
    hypers['n_layers'] = 2
    hypers['dropout'] = 0.4
    hypers['learn_rate'] = 1e-2
    hypers['lr_sched_factor'] = 0.1 
    hypers['lr_sched_patience'] = 5
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT I."
    temperature = 0.1
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

For the same reason we split every dataset to batches.
Training on the whole text will have much higher memory cost, and for large corpus might even be impossible.
Also, by training on sequences (they way we ordered them), makes the model to capture the continuity of the text.

"""

part1_q2 = r"""
**Your answer:**

We initially ordered the data such that following batches will be matched by each sentence (sentence $i$ in batch $j$ will be followed by sentence $i+1$ in batch $j$)
So, during training, the model's "memory" doesn't change between batches, and keeps working on the following sentence.

"""

part1_q3 = r"""
**Your answer:**

For the same reason we mentioned in the previous answer.
The model needs to be fed with continous sentences in each batch entry, thus we can't shuffle them.

"""

part1_q4 = r"""
**Your answer:**

1. A low $T$ will result less uniform distributions -> tends towards argmax behaviour.
Thus the model will be more confident about the next characters while sampling, which means that the more probable outputs will be more likely to be sampled.

2. A high $T$ will result a more uniform distribution, which means that when sampling, the next characters will have a very similar probability to be picked.
Thus, there is a higher chance that we will produce nonesence sentences.

3. As we said before, a low $T$ will result less uniform distributions.
Thus, the sampling will be very close to argmax sampling, which means the outputs will be almost identical to each other (and to the most probable, single sentence that can be produced).


"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
   
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**


"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


# ==============
