import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from evaluate import reconstruction_loss, evaluate_model, visualize_reconstructions, sample_latent_space

class AutoDecoder(nn.Module):
    def __init__(self, latent_dim=128, image_shape=(28, 28), out_channels=1, num_samples=1000):
        super(AutoDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.out_channels = out_channels
        
        self.latents = nn.Parameter(torch.randn(num_samples, latent_dim))
        
        modules = [
            nn.ConvTranspose2d(self.latent_dim, 1024, kernel_size=4, stride=1, padding=0, bias=False),
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
        ]
        
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, indices):
        z = self.latents[indices].unsqueeze(-1).unsqueeze(-1)  # Reshape latent vectors to (batch_size, latent_dim, 1, 1)
        x_rec = self.decoder(z)
        return x_rec.view(len(indices), self.out_channels, self.image_shape[0], self.image_shape[1])
    
    
class AutoDecoderTrainer:
    def __init__(self, model, train_dl, test_dl, device='cuda', learning_rate=0.001):
        self.model = model.to(device)
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    def train(self, num_epochs=20):
        self.model.train()
        for epoch in range(num_epochs):
            train_loss = 0.0
            
            # Wrap the data loader with tqdm for progress bar
            progress_bar = tqdm(self.train_dl, desc=f"Epoch [{epoch+1}/{num_epochs}]")
            
            for indices, images in progress_bar:
                indices = indices.to(self.device)
                images = images.to(self.device)

                self.optimizer.zero_grad()
                
                outputs = self.model(indices)
                
                loss = reconstruction_loss(images, outputs)
                
                loss.backward()
                
                # Gradient Clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                
                # Update progress bar with current loss
                progress_bar.set_postfix({"Loss": f"{train_loss / (progress_bar.n + 1):.4f}"})
                
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(self.train_dl):.4f}')

    def evaluate(self):
        test_loss = evaluate_model(self.model, self.test_dl, self.model.latents, self.device)
        print(f'Evaluation Loss: {test_loss:.4f}')

    def visualize_reconstructions(self):
        visualize_reconstructions(self.model, self.test_dl, self.model.latents, self.device)

    def sample_latent_space(self, n_samples=5):
        sample_latent_space(self.model, n_samples, self.device)
