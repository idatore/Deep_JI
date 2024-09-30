import os
import abc
import sys
import tqdm
import torch
from typing import Any, Callable
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


from cs236781.train_results import FitResult, BatchResult, EpochResult


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, device="cpu"):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
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
        post_epoch_fn=None,
        **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_loss = None
        epochs_without_improvement = 0

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
                
        self.latents = torch.randn(len(dl_train.dataset), self.model.latent_dim, 1, 1, device=self.device)
        
        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)
            
            train_result = self.train_epoch(dl_train, verbose=verbose, **kw)
            test_result = self.test_epoch(dl_test, verbose=verbose, **kw)
            print(f"Test Loss {test_result:.3f}")

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
                    

            # Save model checkpoint if requested
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

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        # return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)
        return (actual_num_epochs, train_loss)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  
        latent_vectors = torch.randn(len(dl_test.dataset), self.model.latent_dim, 1, 1, device=self.device, requires_grad=True)
        latent_optimizer = torch.optim.Adam([latent_vectors], lr=0.01)
        return evaluate_model(self.model, dl_test, latent_optimizer, latent_vectors, 100, self.device)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()


    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
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
            accuracy = 100.0 * num_correct / num_samples
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Accuracy {accuracy:.1f})"
            )

        return EpochResult(losses=losses, accuracy=accuracy)


class RNNTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None):
        super().__init__(model, loss_fn, optimizer, device)

    def train_epoch(self, dl_train: DataLoader, **kw):
        # TODO: Implement modifications to the base method, if needed.
        # ====== YOUR CODE: ======
        
        self.hidden_state = None    
            
        # ========================
        return super().train_epoch(dl_train, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw):
        # TODO: Implement modifications to the base method, if needed.
        # ====== YOUR CODE: ======
        self.hidden_state = None   
        # ========================
        return super().test_epoch(dl_test, **kw)

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)
        seq_len = y.shape[1]

        # TODO:
        #  Train the RNN model on one batch of data.
        #  - Forward pass
        #  - Calculate total loss over sequence
        #  - Backward pass: truncated back-propagation through time
        #  - Update params
        #  - Calculate number of correct char predictions
        # ====== YOUR CODE: ======
        self.optimizer.zero_grad()
        y_pred, self.hidden_state = self.model(x, self.hidden_state)
        self.hidden_state = self.hidden_state.detach()
        y_pred = y_pred.view(-1, y_pred.size(2))  
        y = y.view(-1)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        _, predicted = torch.max(y_pred, 1)
        num_correct = (predicted == y).sum()
        # ========================

        # Note: scaling num_correct by seq_len because each sample has seq_len
        # different predictions.
        return BatchResult(loss.item(), num_correct.item() / seq_len)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)
        seq_len = y.shape[1]

        with torch.no_grad():
            # TODO:
            #  Evaluate the RNN model on one batch of data.
            #  - Forward pass
            #  - Loss calculation
            #  - Calculate number of correct predictions
            # ====== YOUR CODE: ======
            y_pred, _ = self.model(x, self.hidden_state)
            y_pred = y_pred.view(-1, y_pred.size(2)) 
            y = y.view(-1) 
            loss = self.loss_fn(y_pred, y)
            _, predicted = torch.max(y_pred, 1)
            num_correct = (predicted == y).sum()
            # ========================

        return BatchResult(loss.item(), num_correct.item() / seq_len)


def reconstruct_imgs(outputs):
    outputs = (outputs + 1) / 2
    outputs = outputs * 255
    # return outputs.clamp(0, 255).byte()
    return outputs

class DecoderTrainer(Trainer):
        
    def train_batch(self, batch) -> BatchResult:

        indices = batch[0].to(self.device).long() 
        images = batch[1].float().to(self.device)
        images = images / 255.0
        
        for param in self.model.parameters():
            param.requires_grad = True
            
        self.model.train()  
        self.optimizer.zero_grad()

        outputs = self.model.forward(self.latents[indices]).squeeze(1).to(self.device)
        
        # loss = self.loss_fn(reconstruct_imgs(outputs).to(self.device), images)
        loss = self.loss_fn(outputs, images)

        loss.backward()
        self.optimizer.step()

        return BatchResult(loss.item(), 0)

    
    # def test_batch(self, batch) -> BatchResult:
    #     indices = batch[0].to(self.device)  # Batch of sample indices
    #     images = batch[1].float().to(self.device)  # Batch of corresponding images
    #     # images = images / 255.0  # Normalize images to [0, 1] range
    
    #     # Freeze decoder parameters to prevent updating them
    #     for param in self.model.parameters():
    #         param.requires_grad = False
    
    #     # Initialize random latent vectors for test batch
    #     latent_vectors = torch.randn(len(indices), self.model.latent_dim, 1, 1, device=self.device, requires_grad=True)
    
    #     # Define optimizer for latent vectors
    #     latent_optimizer = torch.optim.Adam([latent_vectors], lr=0.01)
    
    #     num_steps = 300 
    #     epochs_without_improvement = 0
    #     best_loss = None
    #     for _ in range(num_steps):
    #         latent_optimizer.zero_grad()
    #         outputs = self.model.decoder(latent_vectors).squeeze(1)
    #         loss = self.loss_fn(outputs, images)
    #         loss.backward()
    #         latent_optimizer.step()
    #         if best_loss is None or loss.item() < best_loss:
    #             best_loss = loss.item()
    #             epochs_without_improvement = 0
    #         else:
    #             epochs_without_improvement += 1
    #             if epochs_without_improvement >= 3:
    #                 break
    
    #     return BatchResult(best_loss, 0) 
    def show_image(self, index):
        """
        Given an index or list of indices, reconstruct the image(s) from the latent vector(s)
        and display them.

        :param index: int or list of ints, index of the latent vector(s)
        """
        # Ensure index is a tensor on the correct device and convert to long
        if isinstance(index, int):
            indices = torch.tensor([index], device=self.device).long()
        else:
            indices = torch.tensor(index, device=self.device).long()

        # Retrieve the latent vector(s)
        z = self.latents[indices]

        # Pass through the decoder (which is self.model in DecoderTrainer)
        with torch.no_grad():
            outputs = self.model(z)

        # Apply sigmoid to bring outputs to [0,1] for visualization (if necessary)
        outputs = torch.sigmoid(outputs)

        # Move outputs to CPU for visualization
        images = outputs.cpu()

        # Plot the image(s)
        if images.shape[0] == 1:
            # If only one image, squeeze out the extra dimension
            plt.imshow(images[0].permute(1, 2, 0).squeeze(), cmap='gray' if images.shape[1] == 1 else None)
            plt.title(f"Reconstructed Image at index {index}")
            plt.axis('off')
            plt.show()
        else:
            # Create a grid of images if there are multiple images to show
            grid_img = make_grid(images, nrow=5, normalize=True, value_range=(0, 1))
            plt.figure(figsize=(12, 6))
            plt.imshow(grid_img.permute(1, 2, 0).numpy(), cmap='gray' if images.shape[1] == 1 else None)
            plt.title("Reconstructed Images")
            plt.axis('off')
            plt.show()

    def evaluate_model_util(test_dl, epochs):

        self.test_latents = nn.Parameter(torch.randn(len(test_dl.dataset), self.model.latent_dim, 1, 1, device=self.device))
        for epoch in range(epochs):
            for i, x in test_dl:
                i = i.to(self.device)
                x = x.to(self.device)
                x_rec = self.model(self.test_latents[i])
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


