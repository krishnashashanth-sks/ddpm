# --- Dummy Data Generation for Demonstration ---
# In a real scenario, X_train_noisy would be your actual noisy images
# and Y_train_noisy would be the target for the NLL loss (the same noisy images).

batch_size = 4
num_samples = 100

# Generate dummy noisy images (input to U-Net)
# Let's assume some base signal and add Gamma noise.
# For a true denoising task, you'd load your actual noisy images here.
# For this demonstration, we'll simulate a noisy input.

# Simulate 'clean' underlying signal (e.g., constant values for simplicity)
clean_signal = np.ones((num_samples, INPUT_H, INPUT_W, INPUT_C)) * 0.5

# Simulate Gamma noise addition
# For simplicity, let's assume noise parameters are fixed for this dummy data
# In reality, the network aims to predict these per-pixel from the noisy input.

dummy_k_true = 2.0  # Example true shape parameter
dummy_theta_true = 0.2 # Example true scale parameter

# Generate noisy data from a Gamma distribution
# np.random.gamma takes shape (k) and scale (theta)
noisy_images = np.random.gamma(shape=dummy_k_true, scale=dummy_theta_true,
                               size=(num_samples, INPUT_H, INPUT_W, INPUT_C))

# Normalize noisy images to a typical image range if necessary (e.g., [0, 1])
noisy_images = noisy_images / np.max(noisy_images) # Simple normalization


# For the gamma_nll_loss, y_true is the observed noisy image itself
# because the loss function evaluates the likelihood of the observation
# given the predicted Gamma parameters.
# So, X_train will be the noisy image, and y_train will also be the noisy image.
# The network learns to predict k, theta such that the likelihood of observing X_train is maximized.

X_train_noisy = noisy_images
Y_train_noisy_target = noisy_images # Target for the NLL loss is the noisy observation itself

print(f"Shape of X_train_noisy: {X_train_noisy.shape}")
print(f"Shape of Y_train_noisy_target: {Y_train_noisy_target.shape}")

