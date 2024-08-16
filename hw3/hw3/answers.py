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
Gradients are maintained during the generator training because its parameters need to be updated based on the discriminator's feedback. They are discarded during the discriminator training since we only need to update the discriminator's parameters, and the generator's output is treated as fixed and shouldn’t be trained here.


"""

part2_q2 = r"""
**Your answer:**
No, you shouldn't stop training a GAN just because the generator loss is low, as this might not mean the generated images are good. They could be repetitive or of poor quality. 
If the discriminator loss stays constant while the generator loss drops, it might indicate that the generator is finding tricks to fool the discriminator without truly improving. This could lead to a situation where the GAN isn't creating diverse and improving images, even though the losses seem to suggest progress.

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
    hypers["embed_dim"] = 120
    hypers["num_heads"] = 6
    hypers["num_layers"] = 2
    hypers["hidden_dim"] = 128
    hypers["window_size"] = 32
    hypers["droupout"] = 0.2
    hypers["lr"] = 5e-4

    # hypers["embed_dim"] = 90
    # hypers["num_heads"] = 6
    # hypers["num_layers"] = 2
    # hypers["hidden_dim"] = 64
    # hypers["window_size"] = 16
    # hypers["droupout"] = 0.1
    # hypers["lr"] = 5e-4

    # # -----1 back-----
    # hypers["embed_dim"] = 128
    # hypers["num_heads"] = 4
    # hypers["num_layers"] = 3
    # hypers["hidden_dim"] = 164
    # hypers["window_size"] = 16
    # hypers["droupout"] = 0.2
    # hypers["lr"] = 4e-4

    # -----2 back-----
    # hypers["embed_dim"] = 128
    # hypers["num_heads"] = 4
    # hypers["num_layers"] = 3
    # hypers["hidden_dim"] = 128
    # hypers["window_size"] = 16
    # hypers["droupout"] = 0.2
    # hypers["lr"] = 2e-4
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**
Stacking encoder layers with sliding-window attention expands the context because each layer focuses on a small part of the input. As more layers are added, the areas that each layer focuses on begin to connect and overlap, allowing the model to see more of the input as a whole. By the final layer, the model has a much wider view of the entire input, similar to how deeper layers in a CNN can capture more detailed features by combining information from earlier layers (based on the hint given).

"""

part3_q2 = r"""
**Your answer:**
We can expand the context while keeping the computational complexity similar by having each token attend to tokens that are spaced apart within its window. Instead of focusing on consecutive neighbors, a token could look at every second token within the window. For example, with a window size of 3, a token at position i could attend to the tokens at positions i-2, i, i+2 instead of i-1, i, i+1.

"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**
The model would likely struggle to fine-tune effectively if the last layers were frozen and only internal layers were fine-tuned. The purpose of the initial layers is to train on general knowledge which will be utilized for the specific task in the later layers. Therefore, the model would be unable to make necessary task-specific adjustments in the later layers that are crucial for optimizing performance on the sentiment analysis task. 

"""


part4_q3= r"""
**Your answer:**
BERT is not directly meant for machine translation because it is an encoder-only model designed for understanding rather than generating sequences. To adapt BERT for machine translation, we would need to integrate it into an encoder-decoder architecture, where BERT serves as the encoder, and a Transformer-based decoder generates the target sequence. The decoder must use masked self-attention to ensure that predictions for the current token depend only on previous tokens and the context from the encoder, not on future tokens. Additionally, pre-training on a seq2seq task, such as translation pairs, would be needed to teach the model how to map input sequences to output sequences effectively. 

"""

part4_q4 = r"""
**Your answer:**
One reason to choose an RNN over a Transformer is its ability to handle tasks requiring strict sequential order, as RNNs process sequences step-by-step, preserving a clear sequence of dependencies between elements. This sequential processing allows RNNs to capture relationships across time steps more naturally, making them particularly effective for tasks like time series analysis or scenarios where the order of inputs significantly influences the output. Additionally, RNNs are simpler and more efficient for smaller datasets or tasks with shorter sequences, where the computational complexity of Transformers may not be necessary.

"""

part4_q5 = r"""
**Your answer:**
NSP in BERT predicts whether a second sentence follows the first in the original text or is randomly selected from the text. This occurs after processing the sentence pair through BERT, with the prediction based on the [CLS] token's representation between sentences. The NSP loss is calculated using cross-entropy.
On the one hand NSP helps BERT understand sentence relationships, which is useful for tasks like question answering or natural language inference. On the other hand, NSP might not be as relevant for different types of tasks, such as text classification or named entity recognition, where the focus is more on individual sentences or words rather than the relationship between sentences. In these cases, the NSP task could be seen as adding unnecessary complexity without providing significant benefits.

"""


# ==============
