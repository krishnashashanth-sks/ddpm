import torch
from model import *
from losses import DenoisingScoreMatchingLoss
import torch.optim as optim
import os
from train import train
from dataset import data_loader,IMG_SIZE
from inference import generate_samples
import torchvision
from torchvision.utils import save_image

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

model = UNet(in_channels=1, out_channels=1, time_emb_dim=256).to(device)

sde = VPSDE(T=1000)

criterion = DenoisingScoreMatchingLoss(sde=sde)

# Hyperparameters
larning_rate = 1e-4
num_epochs = 50

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=larning_rate)

# Create a directory to save generated images
output_dir = 'generated_images'
os.makedirs(output_dir, exist_ok=True)

train(num_epochs,model,data_loader,optimizer,criterion,device)

# Generate samples
num_generate = 64 # Number of images to generate
num_sampling_steps = 250 # Can be less than T for faster generation

generated_images = generate_samples(
    score_model=model,
    sde_model=sde,
    num_samples=num_generate,
    img_size=IMG_SIZE,
    channels=1, # For MNIST, this is 1
    device=device,
    num_sampling_steps=num_sampling_steps
)
# Denormalize images from [-1, 1] to [0, 1] for saving/displaying
generated_images = (generated_images + 1) * 0.5
generated_images = torch.clamp(generated_images, 0, 1) # Ensure values are strictly within [0, 1]

# Save generated images as a grid
grid = torchvision.utils.make_grid(generated_images, nrow=int(math.sqrt(num_generate)))
output_image_path = os.path.join(output_dir, 'generated_samples.png')
save_image(grid, output_image_path)
print(f"Generated {num_generate} images and saved to {output_image_path}")
