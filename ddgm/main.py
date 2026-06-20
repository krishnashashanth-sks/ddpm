from model import build_unet_gamma_estimator
from losses import gamma_nll_loss
import matplotlib.pyplot as plt
from dataset import X_train_noisy,Y_train_noisy_target,batch_size

# 1. Define model input shape
INPUT_H, INPUT_W, INPUT_C = 128, 128, 1
input_shape = (INPUT_H, INPUT_W, INPUT_C)

# 2. Build the model
gamma_unet_model = build_unet_gamma_estimator(input_shape=input_shape)
gamma_unet_model.summary()

# 3. Compile the model with the custom Gamma NLL loss
gamma_unet_model.compile(optimizer='adam', loss=gamma_nll_loss)

# 4. Train the model (conceptual loop)
print("\nStarting conceptual training...")
history = gamma_unet_model.fit(
    X_train_noisy, Y_train_noisy_target,
    batch_size=batch_size,
    epochs=1, # Use a small number of epochs for demonstration
    validation_split=0.2
)

print("\nConceptual training finished.")

# 5. Make a prediction on a dummy noisy image
example_noisy_image = X_train_noisy[0:1]
predicted_gamma_params = gamma_unet_model.predict(example_noisy_image)

print(f"\nShape of predicted Gamma parameters: {predicted_gamma_params.shape}")
print("Example Predicted k (first pixel):", predicted_gamma_params[0, 0, 0, 0])
print("Example Predicted theta (first pixel):", predicted_gamma_params[0, 0, 0, 1])

# Visualize some predictions (optional, for debugging)
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Noisy Input Image')
plt.imshow(example_noisy_image[0, :, :, 0], cmap='gray')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title('Predicted k (Shape Parameter)')
plt.imshow(predicted_gamma_params[0, :, :, 0], cmap='viridis')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title('Predicted theta (Scale Parameter)')
plt.imshow(predicted_gamma_params[0, :, :, 1], cmap='plasma')
plt.colorbar()
plt.tight_layout()
plt.show()