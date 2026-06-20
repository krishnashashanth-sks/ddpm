def train(num_epochs,model,data_loader,optimizer,criterion,device):
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        total_loss = 0
        for batch_idx, (images, _) in enumerate(data_loader):
            images = images.to(device) # Move images to the device

            optimizer.zero_grad() # Zero out the gradients

            loss = criterion(model, images) # Calculate loss
            loss.backward() # Backpropagation
            optimizer.step() # Update model parameters

            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0: # Log every 100 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished, Average Loss: {avg_loss:.4f}")

    print("Training complete!")