import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import numpy as np
import hw3.autoencoder as autoencoder

class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======


        # self.cnn = autoencoder.EncoderCNN(in_size[0], 512)
        # self.modules = [*(self.cnn.layers),
        #                 nn.Flatten(),
        #                 nn.Linear(self._calc_num_cnn_features(in_size), 1)]              
        # self.model = nn.Sequential(*self.modules)
    
        self.discriminator = nn.Sequential(
            nn.Conv2d(self.in_size[0], 128, kernel_size=5, stride=2, padding=2),  
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(1024),

            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0),
        )

        # ========================

    def _calc_num_cnn_features(self, in_shape):
        with torch.no_grad():
            x = torch.zeros(1, *in_shape)
            out_shape = self.cnn(x).shape
        return int(np.prod(out_shape))

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        # y = self.model(x)
        y = self.discriminator(x).view(x.size(0),-1)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim


        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        # self.decoder = autoencoder.DecoderCNN(in_channels=512, out_channels=out_channels)
        # self.modules = [
        #     nn.Linear(z_dim, featuremap_size*featuremap_size*512),
        #     nn.ReLU(True),
        #     nn.Unflatten(-1, (512, featuremap_size, featuremap_size)),
        #     *(self.decoder.layers),
        #     nn.Tanh()]              
        # self.model = nn.Sequential(*self.modules)

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.Tanh() 
        ) 

        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        with torch.set_grad_enabled(with_grad):
            z = torch.randn(n, self.z_dim, 1, 1, device=device)
            samples = self.forward(z)

        # if with_grad:
        #     z = torch.randn(n, self.z_dim, device=device, requires_grad=True)
        #     samples = self.forward(z)
        # else:
        #     with torch.no_grad():
        #         z = torch.randn(n, self.z_dim, device=device)
        #         samples = self.forward(z)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        # z = z.view(z.size(0), -1)
        # x = self.model(z)
        z = z.view(z.size(0), self.z_dim, 1, 1)
        x = self.generator(z)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    # noisy_real_labels = data_label + label_noise * (torch.rand_like(y_data) - 0.5)
    # noisy_fake_labels = (1 - data_label) + label_noise * (torch.rand_like(y_generated) - 0.5)
    # loss_data = F.binary_cross_entropy_with_logits(y_data, noisy_real_labels)
    # loss_generated = F.binary_cross_entropy_with_logits(y_generated, noisy_fake_labels)

    criterion = nn.BCEWithLogitsLoss()

    real_labels = torch.full(y_data.shape, fill_value=data_label, dtype=torch.float, device=y_data.device)
    
    noise_range = (data_label - label_noise / 2, data_label + label_noise / 2)
    real_labels = torch.distributions.Uniform(*noise_range).sample(y_data.shape).to(y_data.device)
    
    loss_data = criterion(y_data, real_labels)
    
    fake_labels = torch.full(y_generated.shape, fill_value=1 - data_label, dtype=torch.float, device=y_generated.device)
    
    noise_range = ((1 - data_label) - label_noise / 2, (1 - data_label) + label_noise / 2)
    fake_labels = torch.distributions.Uniform(*noise_range).sample(y_generated.shape).to(y_generated.device)
    
    loss_generated = criterion(y_generated, fake_labels)

    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    # target_labels = torch.full_like(y_generated, fill_value=data_label, dtype=torch.float)
    # loss = F.binary_cross_entropy_with_logits(y_generated, target_labels)

    criterion = nn.BCEWithLogitsLoss()
    fake_labels = torch.full(y_generated.shape, fill_value=data_label, dtype=torch.float, device=y_generated.device)
    loss = criterion(y_generated, fake_labels)

    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======

    # y_data = dsc_model(x_data)
    # # z = torch.randn(x_data.size(0), gen_model.z_dim, 1, 1, device=x_data.device)
    # # x_generated = gen_model(z)
    # x_generated = gen_model.sample(x_data.size(0))
    # y_generated = dsc_model(x_generated.detach())  
    # dsc_loss = dsc_loss_fn(y_data, y_generated)
    # dsc_loss.backward()
    # dsc_optimizer.step()
    
    fake_img = gen_model.sample(x_data.size(0))
    for _ in range(3):
        dsc_optimizer.zero_grad()
        discr_pred_real = dsc_model(x_data)
        discr_pred_fake = dsc_model(fake_img.detach())
    
        dsc_loss = dsc_loss_fn(discr_pred_real, discr_pred_fake)    
        
        dsc_loss.backward()
        dsc_optimizer.step()

    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    
    for _ in range(3):
        gen_optimizer.zero_grad()
        fake_img = gen_model.sample(x_data.size(0), True)
    
        discr_pred_fake = dsc_model(fake_img)
        gen_loss = gen_loss_fn(discr_pred_fake)
        
        gen_loss.backward()
        gen_optimizer.step()


    # for _ in range(2):
    #     gen_optimizer.zero_grad()
    #     # z = torch.randn(x_data.size(0), gen_model.z_dim, 1, 1, device=x_data.device)
    #     # x_generated = gen_model(z)

    #     x_generated = gen_model.sample(x_data.size(0), True)

    #     y_generated = dsc_model(x_generated)
    #     gen_loss = gen_loss_fn(y_generated)
    #     gen_loss.backward()
    #     gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    if len(gen_losses) <= 5 or gen_losses[-1] > min(gen_losses[-5:-1]):
        return
    # ========================
    torch.save(gen_model, checkpoint_file)
    print(f"*** Saved checkpoint {checkpoint_file} ")
    saved = True
    return saved


