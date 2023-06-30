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
    hypers['batch_size'] = 256
    hypers['seq_len'] = 64
    hypers['h_dim'] = 1024
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.4
    hypers['learn_rate'] = 1e-3
    hypers['lr_sched_factor'] = 0.8 
    hypers['lr_sched_patience'] = 6
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT I."
    temperature = 0.15
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

We initially ordered the data such that following batches will be matched by each sentence (sentence $i$ in batch $j$ will be followed by sentence $i$ in batch $j+1$)
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
    
    hypers['batch_size'] = 64
    hypers['h_dim'] = 128
    hypers['z_dim'] = 32
    hypers['x_sigma2'] = 1e-4
    hypers['learn_rate'] = 0.002
    hypers['betas'] = (0.9, 0.9)
    
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

The $\sigma^2$ hyperparameter is regarded as the variance of the distribution used to model the latent space, where the distribution is multivariate Gaussian, 
parameterized by a mean vector and a covariance matrix. The $\sigma^2$ represents the diagonal elements of the covariance matrix, and it controls the amount of randomness (or diversity) in the generated samples. 
Low values of $\sigma^2$ indicate that the latent space distribution is relatively narrow and that the generated samples are more densed around the mean. Hence, these samples will have less variability and
will be more concentrated around a specific area in the data space. This might provide more deterministic outputs (which are more similar to the inputs) during the generation process.
High values of $\sigma^2$ indicate that the latent space distribution is wider and that the generated samples tend to vary one from another. Here, the generated samples will domenstrate higher variability and randomness during the generation process.
This can lead to higher diversity and to outputs which are less predictable, which means that the VAE will be able to explore more of the lant space regions.
$\sigma^2$ controls the variance of the 

"""

part2_q2 = r"""
**Your answer:**

1. The $\mathcal{L}_{\text{rec}}$ loss purpose is to minimize the difference (the norm) between the original image `x`, and the reconstruced image `xr`.  
The $\mathcal{L}_{\text{KL}}$ loss purpose is to minimize the posterior $p(\bb{Z}|\bb{X})$ so we get closer to the image's distribution.

2. KL-divergence loss term makes sure that the latent space representation will be closer to the prior distribution $p(\bb{Z})$.

3. This effect benefit us by creating seperation between the encoder and the decoder.
The decoder only needs a standard gaussian to be able to decode the image, and the KL-divergence makes sure the encoder's relevant parameters will converge to that gaussian.
By that, we can seperate both parts of the VAE and generate new images with the decoder and some random noise.

"""

part2_q3 = r"""
**Your answer:**

By maximizing the evidence distribution $p(\bb{X})$, the VAE learns to capture the underlying structure of the data and generate meaningful samples from the learned latent space.
This approach allows the VAE to generate new samples by sampling from the latent space distribution and passing them through the decoder.
"""

part2_q4 = r"""
**Your answer:**

We use the log variance for numerical stability.
$\sigma$ can be very low and might cause numerical issues, using the log-space helps reduce these issues.
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
        lr = 0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    hypers['embed_dim'] = 256
    hypers['num_heads'] = 4
    hypers['num_layers'] = 6
    hypers['hidden_dim'] = 64
    hypers['window_size'] = 16
    hypers['droupout'] = 0.1
    hypers['lr'] = 1e-4
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**
Similar to how the respective field works in CNNs, stacking encoder layers on top of each other creates an "attention field-of-view" effect.
On the lowest level, each value in the output is affected by keys in the window around the respective query.
On the second level, the input is the output of the previous layer, thus already affected by the previous window, and now the window gets wider by 2 elements (the outliers of the window)
Thus, on the last layer, the effective window size is $window\_size + 2 \cdot \# layers$.
"""

part3_q2 = r"""
**Your answer:**

We propose a variation described in the paper.
The proposition is called "Global Attention".
the idea is to combine the regular sliding window with some constant "global keys" - a few pre-selected (and task-specific) input locations.
We make this attention operation symmetric - a token with a global attention attends to all tokens across the sequence, and all tokens in the sequence attend to it.
These input locations are task-specific, for example - for classification, the global attention is used on the [CLS] location.
Since the number of such tokens is small relative to $n$ the complexity of the combined local and global attention is still $\mathcal{O}(wn)$.

"""


part4_q1 = r"""
**Your answer:**

The obtained results demonstrate a fairly good performance (around 80%), which in comparison to the results of the previous part - are significantly better.
We believe that this performance difference stems from the fact that the fine-tuned model was originally train on a much larger dataset, and also its fundamental
architecture is different in aspects of task compatibility. This might not always be the case on any downstream task, where if the fine-tuned model was originally trained on
a relatively small dataset which doesn't represent the data's underlying distribution, or if its architecture was not suitable for a given task - the results we get would be worse 
than those of a "trained from scratch" encoder.    
"""

part4_q2 = r"""
**Your answer:**

In this case the model's results would be worse, since un-freezing and modifying internal model layers (such as the attention blocks) could potentially disrupt the model's ability to capture contextual
information effectively - leading to less accurate results. The last two linear layers, on the other hand, typically contain task-specific information and fine-tuning only them allows the model to adapt to the target task more accurately.
"""


# ==============
