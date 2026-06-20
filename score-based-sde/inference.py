import math
import torch

@torch.no_grad()
def generate_samples(score_model, sde_model, num_samples, img_size, channels, device, num_sampling_steps=None, z_initial=None):
    score_model.eval()
    if num_sampling_steps is None:
        num_sampling_steps = sde_model.T # Default to T steps

    # Start from pure noise at T
    if z_initial is None:
      x = torch.randn(num_samples, channels, img_size, img_size, device=device)
    else:
      x = z_initial.to(device)

    # Time points for sampling (from T down to a small epsilon, inclusive)
    # The sde_model expects `t` in [0, T] range. We generate `num_sampling_steps` transitions.
    time_points = torch.linspace(sde_model.T, sde_model.eps, num_sampling_steps + 1).to(device)

    for i in range(num_sampling_steps):
        t_curr = time_points[i] # Current time point for x_t (e.g., T, T - dt, ...)
        t_prev = time_points[i+1] # Previous time point for x_{t-1} (t_curr - dt)

        # Create batch-wise time tensor for the model prediction
        t_curr_batch = torch.full((num_samples,), t_curr.item(), device=device)

        # Predict noise using the score model
        predicted_noise = score_model(x, t_curr_batch)

        # Get SDE parameters for current and previous time steps
        beta_t = sde_model.beta(t_curr_batch) # beta(t)
        alpha_t = 1.0 - beta_t # alpha(t)
        alpha_t_bar = sde_model.alpha_sq_cum(t_curr_batch) # alpha_bar(t)

        # For the previous time point, t_prev, used to get alpha_bar(t-1)
        # It's important that t_prev is also within the SDE's `T` range. `sde_model.eps` should be > 0
        t_prev_batch = torch.full((num_samples,), t_prev.item(), device=device)
        alpha_t_prev_bar = sde_model.alpha_sq_cum(t_prev_batch) # alpha_bar(t-1)

        # Reshape for broadcasting with image tensor (B, C, H, W)
        beta_t_reshaped = beta_t.view(-1, 1, 1, 1)
        alpha_t_reshaped = alpha_t.view(-1, 1, 1, 1)
        alpha_t_bar_reshaped = alpha_t_bar.view(-1, 1, 1, 1)
        alpha_t_prev_bar_reshaped = alpha_t_prev_bar.view(-1, 1, 1, 1)

        # Calculate x_0_hat (estimated original data point) based on x_t and predicted_noise
        # x_0_hat = (x - sqrt(1 - alpha_bar(t)) * epsilon_theta) / sqrt(alpha_bar(t))
        x_0_hat = (x - torch.sqrt(1.0 - alpha_t_bar_reshaped) * predicted_noise) / torch.sqrt(alpha_t_bar_reshaped)

        # Calculate the mean for the reverse step, mu(x_t, x_0_hat)
        # mu_tilde(x_t, x_0) = (sqrt(alpha_bar(t-1)) * beta(t) * x_0 + sqrt(alpha(t)) * (1 - alpha_bar(t-1)) * x_t) / (1 - alpha_bar(t))
        mean = (torch.sqrt(alpha_t_prev_bar_reshaped) * beta_t_reshaped * x_0_hat + \
                torch.sqrt(alpha_t_reshaped) * (1.0 - alpha_t_prev_bar_reshaped) * x) / \
               (1.0 - alpha_t_bar_reshaped)

        # Calculate the variance for the reverse step, sigma_t_sq
        # sigma_tilde_sq(t) = beta(t) * (1 - alpha_bar(t-1)) / (1 - alpha_bar(t))
        variance = beta_t_reshaped * (1.0 - alpha_t_prev_bar_reshaped) / (1.0 - alpha_t_bar_reshaped)
        log_variance = torch.log(variance.clamp(min=1e-20)) # Clamp for numerical stability

        # Sample z for the stochastic term, unless it's the last step (deterministic for x_0)
        if i == num_sampling_steps - 1:
            x = mean
        else:
            z = torch.randn_like(x)
            # Sample from N(mean, variance)
            x = mean + torch.exp(0.5 * log_variance) * z

        # Optional: clamp to [-1, 1] range after each step to prevent values from exploding
        x = torch.clamp(x, -1., 1.)

    return x