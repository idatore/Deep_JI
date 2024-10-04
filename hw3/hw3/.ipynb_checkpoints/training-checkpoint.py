import os
import abc
import sys
import tqdm
import torch
import torch.nn as nn
from typing import Any, Callable
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


from cs236781.train_results import FitResult, BatchResult, EpochResult


class DecoderTrainer(abc.ABC):

    def __init__(self, model, loss_fn, optimizer, device="cpu"):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        model.to(self.device)

    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every=1,
        **kw,
    ) -> FitResult:
        self.num_epochs = num_epochs
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_loss = None
        epochs_without_improvement = 0

        self.latents = torch.randn(len(dl_train.dataset), self.model.latent_dim, 1, 1, device=self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.model.decoder.parameters(), 'lr': 0.001},
            {'params': [self.model.mu, self.model.sigma], 'lr': 0.01}
        ])

        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f"{checkpoints}.pt"
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f"*** Loading checkpoint file {checkpoint_filename}")
                saved_state = torch.load(checkpoint_filename, map_location=self.device)
                best_loss = saved_state.get("best_loss", best_loss)
                epochs_without_improvement = saved_state.get(
                    "ewi", epochs_without_improvement
                )
                self.model.load_state_dict(saved_state["model_state"])

        for epoch in range(num_epochs):
            self.epoch = epoch
            save_checkpoint = False
            verbose = False 
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)
            train_result = self.train_epoch(dl_train, verbose=verbose, **kw)
            test_result = 0
            # test_result = self.test_epoch(dl_test, verbose=verbose, **kw)
            # print(f"Test Loss {test_result:.3f}")

            train_loss += train_result.losses

            actual_num_epochs += 1
            if best_loss is None or test_result < best_loss:
                save_checkpoint = True
                best_loss = test_result
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if early_stopping and epochs_without_improvement >= early_stopping:
                    break
                    

            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(
                    best_loss=best_loss,
                    ewi=epochs_without_improvement,
                    model_state=self.model.state_dict(),
                )
                torch.save(saved_state, checkpoint_filename)
                print(
                    f"*** Saved checkpoint {checkpoint_filename} " f"at epoch {epoch+1}"
                )

        return (actual_num_epochs, train_loss)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        self.model.train(True)
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        self.model.train(False)  
        latent_vectors = torch.randn(len(dl_test.dataset), self.model.latent_dim, 1, 1, device=self.device, requires_grad=True)
        latent_optimizer = torch.optim.Adam([latent_vectors], lr=0.1)
        return evaluate_model(self.model, dl_test, latent_optimizer, latent_vectors, 25, self.device)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        raise NotImplementedError()


    @staticmethod
    def _print(message, verbose=True):
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
    ) -> EpochResult:
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}"
            )
        accuracy = 0
        return EpochResult(losses=losses, accuracy=accuracy)


class AutoDecoderTrainer(DecoderTrainer):
        
    def train_batch(self, batch) -> BatchResult:

        indices = batch[0].to(self.device).long() 
        images = batch[1].float().to(self.device)
        
        self.model.train()  
        self.optimizer.zero_grad()

        outputs = self.model.forward(self.latents[indices]).squeeze(1).to(self.device)
        
        loss = self.loss_fn(outputs, images)

        loss.backward()
        self.optimizer.step()

        return BatchResult(loss.item(), 0)

    def show_image(self, index):
        if isinstance(index, int):
            indices = torch.tensor([index], device=self.device).long()
        else:
            indices = torch.tensor(index, device=self.device).long()
    
        z = self.latents[indices]
    
        with torch.no_grad():
            outputs = self.model(z)
    
        images = outputs.cpu()
    
        if images.shape[0] == 1:
            plt.imshow(images[0], cmap='gray')
        else:
            grid_img = make_grid(images.unsqueeze(1), nrow=5, normalize=True, value_range=(0, 1))
            plt.figure(figsize=(12, 6))
            plt.imshow(grid_img.permute(1, 2, 0).numpy(), cmap='gray')

    
        plt.title(f"Reconstructed Image(s) at index {index}")
        plt.axis('off')
        plt.show()



class VAD_Trainer(DecoderTrainer):
    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every=1,
        **kw,
    ) -> FitResult:

        self.labels = dl_train.dataset.y.to(self.device)

        return super().fit(dl_train, dl_test, num_epochs, checkpoints, early_stopping, print_every, **kw)

    def train_batch(self, batch) -> BatchResult:
        
        indices = batch[0].long().to(self.device)
        images = batch[1].float().to(self.device)
        labels = self.labels[indices].to(self.device)

        self.model.train()  
        self.optimizer.zero_grad()

        latents = self.model.reparameterize(self.latents[indices], labels).to(self.device)
        outputs = self.model(latents).squeeze(1).to(self.device)

        sigma_sq = self.model.sigma[labels].pow(2)
        sigma_sq = torch.clamp(sigma_sq, min=1e-6)
        mu_sq = self.model.mu[labels].pow(2)
        # beta = min(1.0, (self.epoch + 1) / self.num_epochs)
        beta = 0
        loss = self.loss_fn(outputs, images, mu_sq, sigma_sq, beta)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        return BatchResult(loss.item(), 0)

    def show_image(self, index):
        if isinstance(index, int):
            indices = torch.tensor([index], device=self.device).long()
        else:
            indices = torch.tensor(index, device=self.device).long()
            
        labels = self.labels[indices]
        z = self.model.reparameterize(self.latents[indices], labels)
    
        with torch.no_grad():
            outputs = self.model(z)
    
        images = outputs.cpu()
    
        if images.shape[0] == 1:
            plt.imshow(images[0], cmap='gray')
        else:
            grid_img = make_grid(images.unsqueeze(1), nrow=5, normalize=True, value_range=(0, 1))
            plt.figure(figsize=(12, 6))
            plt.imshow(grid_img.permute(1, 2, 0).numpy(), cmap='gray')

    
        plt.title(f"Reconstructed Image(s) at index {index}")
        plt.axis('off')
        plt.show()

    def show_image_sample(self, label):
    
        z = self.model.sample(label)
    
        with torch.no_grad():
            outputs = self.model(z)
    
        images = outputs.cpu()
    
        if images.shape[0] == 1:
            plt.imshow(images[0], cmap='gray')
        else:
            grid_img = make_grid(images.unsqueeze(1), nrow=5, normalize=True, value_range=(0, 1))
            plt.figure(figsize=(12, 6))
            plt.imshow(grid_img.permute(1, 2, 0).numpy(), cmap='gray')

    
        plt.title(f"Sampled Image(s) of class {label}")
        plt.axis('off')
        plt.show()


def reconstruction_loss(x, x_rec):
    """
    :param x: the original images
    :param x_rec: the reconstructed images
    :return: the reconstruction loss
    """
    return torch.norm(x - x_rec) / torch.prod(torch.tensor(x.shape))
    
def evaluate_model(model, test_dl, opt, latents, epochs, device):
    """
    :param model: the trained model
    :param test_dl: a DataLoader of the test set
    :param opt: a torch.optim object that optimizes ONLY the test set
    :param latents: initial values for the latents of the test set
    :param epochs: how many epochs to train the test set latents for
    :return:
    """
    for epoch in range(epochs):
        for i, x in test_dl:
            i = i.to(device)
            x = x.to(device)
            x_rec = model(latents[i])
            loss = reconstruction_loss(x, x_rec)
            opt.zero_grad()
            loss.backward()
            opt.step()

    losses = []
    with torch.no_grad():
        for i, x in test_dl:
            i = i.to(device)
            x = x.to(device)
            x_rec = model(latents[i])
            loss = reconstruction_loss(x, x_rec)
            losses.append(loss.item())

        final_loss = sum(losses) / len(losses)

    return final_loss



