import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from evaluate import reconstruction_loss, evaluate_model
import matplotlib.pyplot as plt

class AutoDecoder(nn.Module):
    def __init__(self, latent_dim=64, num_samples=1000):
        super(AutoDecoder, self).__init__()
        self.latent_dim = latent_dim
        
        # self.latents = nn.Parameter(torch.randn(num_samples, latent_dim) * 0.1)
        self.latents = torch.randn(num_samples, latent_dim) * 0.1
        
        self.fc = nn.Linear(latent_dim, 7 * 7 * 64)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 7, 7)
        x = self.deconv(x)
        return x

class AutoDecoderTrainer:
    def __init__(self, model, train_dl, test_dl, device='cuda', learning_rate=0.0005):
        self.model = model.to(device)
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.device = device
        
        self.optimizer = optim.Adam(list(self.model.parameters()), lr=learning_rate)
        
    def train(self, num_epochs=20):
        self.model.train()
        for epoch in range(num_epochs):
            self.show_first_reconstructed_image()
            train_loss = 0.0
            progress_bar = tqdm(self.train_dl, desc=f"Epoch [{epoch+1}/{num_epochs}]")
            
            for indices, images in progress_bar:
                indices = indices.to(self.device)
                images = images.to(self.device)

                self.optimizer.zero_grad()
                
                outputs = self.model.forward(self.model.latents[indices].to(device))
                
                loss = reconstruction_loss(images, outputs)
                
                loss.backward()
                
                # Gradient Clipping to prevent exploding gradients
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                
                # Update progress bar with current loss
                progress_bar.set_postfix({"Loss": f"{train_loss / (progress_bar.n + 1):.4f}"})
                        
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(self.train_dl):.4f}')

    def show_first_reconstructed_image(self):
        """
        Reconstruct and display the first image from the first latent vector in self.model.latents
        """
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Get the first latent vector
            first_latent = self.model.latents[0].unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Pass the latent vector through the model to get the reconstructed image
            reconstructed_image = self.model(first_latent)  # The model is the decoder
            
            # Convert the tensor to numpy for display
            image = reconstructed_image.squeeze().cpu().numpy()  # Remove batch and channel dimensions
            
            # Display the reconstructed image using matplotlib
            plt.imshow(image, cmap='gray')  # Assuming grayscale images
            plt.title('Reconstructed Image from First Latent Vector')
            plt.axis('off')
            plt.show()



    def evaluate(self, epochs):
        test_loss = evaluate_model(model=self.model, test_dl=self.test_dl, opt=self.optimizer, latents=self.model.latents, epochs=epochs, device=self.device)
        print(f'Evaluation Loss: {test_loss:.4f}')


    
