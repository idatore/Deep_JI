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
    hypers["batch_size"] = 128
    hypers["seq_len"] = 25
    hypers["h_dim"] = 256
    hypers["n_layers"] = 3
    hypers["dropout"] = 0.1
    hypers["learn_rate"] = 0.002
    hypers["lr_sched_factor"] = 0.1
    hypers["lr_sched_patience"] = 3
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "Act I:"
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
Splitting the corpus into sequences instead of training on the entire text is crucial for memory efficiency and effective learning. Processing the entire text at once would require a vast amount of memory, especially for longer texts, which may not be feasible on typical hardware. By dividing the text into smaller sequences, the model can process these chunks more easily, fitting within memory constraints and enabling batch processing. This approach also helps in stabilizing gradient updates, making the training process more reliable.

Moreover, splitting the text into sequences enhances learning by allowing the model to focus on short term dependencies within each chunk. This is particularly important when using BPTT, as it allows the model to learn temporal dependencies over these sequences effectively. BPTT works well within these smaller chunks, making the optimization process more manageable and computationally feasible.
"""

part1_q2 = r"""

The generated text can show memory longer than the sequence length because RNNs maintain a hidden state that carries information across time steps. This hidden state enables the model to retain context from previous sequences, even when the current sequence is short. Through BPTT, the model learns to use this hidden state to capture dependencies across multiple sequences, allowing it to generate text that reflects earlier parts of the corpus.
"""

part1_q3 = r"""
We do not shuffle the order of batches when training because maintaining the sequential order of data is crucial for RNN. This model rely on the continuity of the input data to capture temporal dependencies and maintain the context across sequences. Shuffling would disrupt this order, preventing the model from learning meaningful patterns and dependencies within the data.

"""

part1_q4 = r"""
1. Lowering the temperature makes the model's predictions more confident by sharpening the probability distribution. It reduces the likelihood of sampling less probable outcomes, leading to more deterministic and focused text generation.

2. When the temperature is very high, the probability distribution becomes more uniform, making the model more likely to sample from all possible outcomes, including those with low probability. This can result in more random and less coherent text.

3. When the temperature is very low, the probability distribution becomes more peaked, making the model almost always select the most likely outcome. This can lead to repetitive and predictable text, as the model is less likely to explore diverse options


"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None

def part3_gan_hyperparams():
    hypers = dict()
    hypers["batch_size"] = 30  
    hypers["z_dim"] = 100
    hypers["discriminator_optimizer"] = {
        "type": "Adam", 
        "lr": 0.00015,

        "betas": (0.5, 0.999)  
    }
    hypers["generator_optimizer"] = {
        "type": "Adam", 
        "lr": 0.00015,
        "betas": (0.5, 0.999),
    }
    
    hypers["data_label"] = 0
    hypers["label_noise"] = 0.01

    return hypers

def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.1
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.5, 0.999)
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

    #todo jonah - update to work
    hypers["embed_dim"] = 256
    hypers["num_heads"] = 4
    hypers["num_layers"] = 6
    hypers["hidden_dim"] = 64
    hypers["window_size"] = 16
    hypers["dropout"] = 0.2
    hypers["lr"] = 1e-4
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


part4_q3= r"""
**Your answer:**


"""

part4_q4 = r"""
**Your answer:**


"""

part4_q5 = r"""
**Your answer:**


"""


# ==============
