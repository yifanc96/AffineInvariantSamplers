### not affine invariant yet

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from typing import Callable, Tuple

def ensemble_bps(grad_log_prob, initial, n_samples, n_walkers=20, dt=0.01, refresh_rate=1.0, n_thin=1):
    """
    Vectorized implementation of Ensemble Bouncy Particle Sampler with covariance preconditioning.
    
    Parameters:
    -----------
    grad_log_prob : function
        Gradient of log probability density function that accepts array of shape (n_walkers, dim)
        and returns array of shape (n_walkers, dim)
    initial : array
        Initial state (will be used as mean for initializing walkers)
    n_samples : int
        Number of samples to draw per walker
    n_walkers : int
        Number of walkers in the ensemble (must be even)
    dt : float
        Time step for integration
    refresh_rate : float
        Rate parameter for velocity refreshment
    n_thin : int
        Thinning factor - store every n_thin sample (default: 1, no thinning)
        
    Returns:
    --------
    samples : array
        Samples from all walkers (shape: n_walkers, n_samples, dim)
    diagnostics : dict
        Dictionary containing bounce rates, refresh rates, and other diagnostics
    """
    # Ensure even number of walkers
    if n_walkers % 2 != 0:
        n_walkers += 1
        
    dim = len(initial)
    half_walkers = n_walkers // 2
    
    # Initialize walkers with small random perturbations around initial
    walkers = np.tile(initial, (n_walkers, 1)) + 0.1 * np.random.randn(n_walkers, dim)
    
    # Initialize velocities on unit sphere
    velocities = np.random.randn(n_walkers, dim)
    velocity_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    velocities = velocities / velocity_norms
    
    # Calculate total iterations needed based on thinning factor
    total_iterations = n_samples * n_thin
    
    # Storage for samples and tracking events
    samples = np.zeros((n_walkers, n_samples, dim))
    n_bounces = np.zeros(n_walkers)
    n_refreshes = np.zeros(n_walkers)
    
    # Sample index to track where to store thinned samples
    sample_idx = 0
    
    # Main sampling loop
    for i in range(total_iterations):
        # Store current state from all walkers (only every n_thin iterations)
        if i % n_thin == 0 and sample_idx < n_samples:
            samples[:, sample_idx] = walkers
            sample_idx += 1
        
        # Integrate positions for all walkers
        walkers += dt * velocities
        
        # Process each half of the ensemble (complementary preconditioning)
        for half in [0, 1]:
            # Set indices for active and complementary walker sets
            active_indices = np.arange(half * half_walkers, (half + 1) * half_walkers)
            comp_indices = np.arange((1 - half) * half_walkers, (2 - half) * half_walkers)
            
            # Compute empirical covariance from complementary ensemble
            comp_positions = walkers[comp_indices]
            if len(comp_indices) > 1:
                cov_matrix = np.cov(comp_positions.T) + 1e-6 * np.eye(dim)
                if cov_matrix.ndim == 0:  # Handle 1D case
                    cov_matrix = np.array([[cov_matrix]])
            else:
                cov_matrix = np.eye(dim)
            
            # Compute gradients for active walkers
            active_walkers = walkers[active_indices]
            gradients = grad_log_prob(active_walkers)
            
            # Apply covariance preconditioning: C * gradient
            preconditioned_grads = gradients @ cov_matrix.T
            
            # Compute bounce rates: max(0, -v^T * preconditioned_gradient)
            active_velocities = velocities[active_indices]
            dot_products = np.sum(active_velocities * preconditioned_grads, axis=1)
            bounce_rates = np.maximum(0.0, -dot_products)
            
            # Determine bounces using Poisson process approximation
            bounce_probs = bounce_rates * dt
            bounce_decisions = np.random.uniform(size=half_walkers) < bounce_probs
            
            if np.any(bounce_decisions):
                # Get bouncing walkers
                bouncing_indices = active_indices[bounce_decisions]
                bouncing_grads = preconditioned_grads[bounce_decisions]
                bouncing_velocities = velocities[bouncing_indices]
                
                # Normalize gradients for reflection
                grad_norms = np.linalg.norm(bouncing_grads, axis=1, keepdims=True)
                valid_grad_mask = grad_norms.flatten() > 1e-10
                
                if np.any(valid_grad_mask):
                    # Compute reflection normals
                    normals = np.zeros_like(bouncing_grads)
                    normals[valid_grad_mask] = (bouncing_grads[valid_grad_mask] / 
                                              grad_norms[valid_grad_mask])
                    
                    # Vectorized reflection: v' = v - 2(v·n)n
                    dot_vn = np.sum(bouncing_velocities * normals, axis=1, keepdims=True)
                    reflected_velocities = bouncing_velocities - 2 * dot_vn * normals
                    
                    # Update velocities
                    velocities[bouncing_indices] = reflected_velocities
                    
                    # Track bounces
                    n_bounces[bouncing_indices] += 1
            
            # Handle velocity refreshment
            refresh_probs = refresh_rate * dt
            refresh_decisions = np.random.uniform(size=half_walkers) < refresh_probs
            
            if np.any(refresh_decisions):
                refreshing_indices = active_indices[refresh_decisions]
                n_refreshing = len(refreshing_indices)
                
                # Generate new random velocities on unit sphere
                new_velocities = np.random.randn(n_refreshing, dim)
                new_norms = np.linalg.norm(new_velocities, axis=1, keepdims=True)
                new_velocities = new_velocities / new_norms
                
                # Update velocities
                velocities[refreshing_indices] = new_velocities
                
                # Track refreshes
                n_refreshes[refreshing_indices] += 1
    
    # Calculate diagnostics
    bounce_rates = n_bounces / total_iterations
    refresh_rates = n_refreshes / total_iterations
    
    diagnostics = {
        'n_bounces': n_bounces,
        'n_refreshes': n_refreshes,
        'bounce_rates': bounce_rates,
        'refresh_rates': refresh_rates,
        'mean_bounce_rate': np.mean(bounce_rates),
        'mean_refresh_rate': np.mean(refresh_rates)
    }
    
    return samples, diagnostics


def example_usage():
    """
    Example usage with a 2D Gaussian mixture target distribution.
    """
    # Define gradient of log probability for mixture of Gaussians
    def grad_log_prob(x):
        """
        Vectorized gradient computation for mixture of two Gaussians.
        Input: x of shape (n_walkers, 2)
        Output: gradients of shape (n_walkers, 2)
        """
        mu1, mu2 = np.array([-2, -2]), np.array([2, 2])
        cov1, cov2 = np.eye(2), np.eye(2)
        
        # Compute probabilities for all walkers
        diff1 = x - mu1[np.newaxis, :]
        diff2 = x - mu2[np.newaxis, :]
        
        p1 = np.exp(-0.5 * np.sum(diff1**2, axis=1))
        p2 = np.exp(-0.5 * np.sum(diff2**2, axis=1))
        
        # Compute gradients
        grad1 = -diff1  # gradient of -0.5 * ||x - mu1||^2
        grad2 = -diff2  # gradient of -0.5 * ||x - mu2||^2
        
        # Weighted combination
        total_prob = 0.5 * (p1 + p2) + 1e-10
        weights1 = (0.5 * p1 / total_prob)[:, np.newaxis]
        weights2 = (0.5 * p2 / total_prob)[:, np.newaxis]
        
        return weights1 * grad1 + weights2 * grad2
    
    # Run ensemble BPS
    np.random.seed(42)
    initial = np.array([0.0, 0.0])
    
    print("Running Ensemble BPS...")
    samples, diagnostics = ensemble_bps(
        grad_log_prob=grad_log_prob,
        initial=initial,
        n_samples=1000,
        n_walkers=20,
        dt=0.05,
        refresh_rate=2.0,
        n_thin=5
    )
    
    print(f"Sample shape: {samples.shape}")
    print(f"Mean bounce rate: {diagnostics['mean_bounce_rate']:.3f}")
    print(f"Mean refresh rate: {diagnostics['mean_refresh_rate']:.3f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot trajectories
    axes[0, 0].set_title("Walker Trajectories")
    for walker in range(samples.shape[0]):
        walker_samples = samples[walker]
        axes[0, 0].plot(walker_samples[:, 0], walker_samples[:, 1], 
                       alpha=0.6, linewidth=0.8)
    axes[0, 0].set_xlabel("X1")
    axes[0, 0].set_ylabel("X2")
    axes[0, 0].grid(True)
    
    # Plot final positions
    final_positions = samples[:, -1, :]
    half_walkers = samples.shape[0] // 2
    
    axes[0, 1].set_title("Final Positions (Group 1: blue, Group 2: red)")
    axes[0, 1].scatter(final_positions[:half_walkers, 0], 
                      final_positions[:half_walkers, 1], 
                      c='blue', alpha=0.7, label='Group 1')
    axes[0, 1].scatter(final_positions[half_walkers:, 0], 
                      final_positions[half_walkers:, 1], 
                      c='red', alpha=0.7, label='Group 2')
    axes[0, 1].set_xlabel("X1")
    axes[0, 1].set_ylabel("X2")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot marginal distributions
    all_samples = samples.reshape(-1, 2)
    axes[1, 0].hist(all_samples[:, 0], bins=50, alpha=0.7, density=True)
    axes[1, 0].set_title("Marginal Distribution X1")
    axes[1, 0].set_xlabel("X1")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].grid(True)
    
    axes[1, 1].hist(all_samples[:, 1], bins=50, alpha=0.7, density=True)
    axes[1, 1].set_title("Marginal Distribution X2")
    axes[1, 1].set_xlabel("X2")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return samples, diagnostics

if __name__ == "__main__":
    samples, diagnostics = example_usage()