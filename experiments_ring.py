import numpy as np
import matplotlib.pyplot as plt
import time

from samplers import side_move, stretch_move, hmc, hamiltonian_walk_move, hamiltonian_side_move
from autocorrelation_func import autocorrelation_fft, integrated_autocorr_time


def benchmark_samplers_ring(dim=10, n_samples=10000, burn_in=1000, sigma=0.1):
    """
    Benchmark different MCMC samplers on a ring-shaped distribution.
    
    Parameters:
    -----------
    dim : int
        Dimension of the distribution
    n_samples : int
        Number of samples to generate after burn-in
    burn_in : int
        Number of initial samples to discard as burn-in
    sigma : float
        Width parameter of the ring (smaller values make a sharper ring)
    """
    # Define the ring distribution log-density
    def log_density(x):
        """Log density of the ring-shaped distribution"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Calculate squared radius for each sample
        radius_sq = np.sum(x**2, axis=1)
        
        # Calculate potential: (r^2 - 1)^2 / sigma^2
        potential = (radius_sq - 1.0)**2 / (sigma**2)
        
        # Log density is negative potential (up to normalization constant)
        return -potential
    
    # Define the gradient of the negative log density
    def gradient(x):
        """Gradient of the negative log density (potential gradient)"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        batch_size = x.shape[0]
        grad = np.zeros_like(x)
        
        # Calculate the squared radius for each sample
        radius_sq = np.sum(x**2, axis=1, keepdims=True)  # Shape: (batch_size, 1)
        
        # Calculate the gradient formula: 4(r^2-1)x / sigma^2
        # The factor of 4 comes from the chain rule derivative
        grad = 4.0 * (radius_sq - 1.0) * x / (sigma**2)
        
        # Return negative gradient (for potential)
        return grad
    
    # Define the potential energy (negative log density)
    def potential(x):
        """Negative log density (potential energy)"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Calculate squared radius for each sample
        radius_sq = np.sum(x**2, axis=1)
        
        # Calculate potential energy: (r^2 - 1)^2 / sigma^2
        return (radius_sq - 1.0)**2 / (sigma**2)
    
    # Initial state - place points near but not exactly on the ring
    initial = np.random.randn(dim)
    initial = initial / np.sqrt(np.sum(initial**2)) * 1.2  # Start slightly outside the ring
    
    # Dictionary to store results
    results = {}
    
    # Define samplers to benchmark with burn-in
    total_samples = n_samples + burn_in
    
    # Define samplers to benchmark - parameters tuned for the ring distribution
    samplers = {
        "Side Move": lambda: side_move(log_density, initial, total_samples, n_walkers=dim*2, gamma=1.687),
        "Stretch Move": lambda: stretch_move(log_density, initial, total_samples, n_walkers=dim*2, a=1.0+2.151/np.sqrt(dim)),
        "HMC n=10": lambda: hmc(log_density, initial, total_samples, gradient, epsilon=0.1, L=10, n_chains=1),
        "HMC n=2": lambda: hmc(log_density, initial, total_samples, gradient, epsilon=0.5, L=2, n_chains=1),
        "Hamiltonian Walk Move n=10": lambda: hamiltonian_walk_move(gradient, potential, initial, total_samples, 
                                                        n_chains_per_group=dim, epsilon=0.1, n_leapfrog=10, beta=1.0),
        "Hamiltonian Walk Move n=2": lambda: hamiltonian_walk_move(gradient, potential, initial, total_samples, 
                                                        n_chains_per_group=dim, epsilon=0.5, n_leapfrog=2, beta=1.0),
        "Hamitonian Side Move n=10": lambda: hamiltonian_side_move(gradient, potential, initial, total_samples,
                                                        n_chains_per_group=dim, epsilon=0.1, n_leapfrog=10, beta=1.0),
        "Hamitonian Side Move n=2": lambda: hamiltonian_side_move(gradient, potential, initial, total_samples, 
                                                                              n_chains_per_group=dim, epsilon=0.5, n_leapfrog=2, beta=1.0),
    }
    
    # Benchmark each sampler with careful error handling
    for name, sampler_func in samplers.items():
        print(f"Running {name}...")
        start_time = time.time()
        
        try:
            samples, acceptance_rates = sampler_func()
            
            # Apply burn-in: discard the first burn_in samples
            post_burn_in_samples = samples[:, burn_in:, :]
            
            # Flatten samples from all chains
            flat_samples = post_burn_in_samples.reshape(-1, dim)
            
            # Calculate mean distance from ring
            radius = np.sqrt(np.sum(flat_samples**2, axis=1))
            mean_radius = np.mean(radius)
            radius_std = np.std(radius)
            mean_distance_from_ring = np.mean(np.abs(radius - 1.0))
            
            # Compute autocorrelation for first dimension
            acf = autocorrelation_fft(np.mean(post_burn_in_samples[:, :, 0],axis=0))
            
            # Compute integrated autocorrelation time for first dimension
            try:
                tau, _, ess = integrated_autocorr_time(np.mean(post_burn_in_samples[:, :, 0],axis=0))
            except:
                tau, ess = np.nan, np.nan
                print("  Warning: Could not compute integrated autocorrelation time")
            
        except Exception as e:
            print(f"  Error with {name}: {str(e)}")
            # Create dummy data in case of error
            flat_samples = np.zeros((10, dim))
            acceptance_rates = np.zeros(dim)
            mean_radius = np.nan
            radius_std = np.nan
            mean_distance_from_ring = np.nan
            acf = np.zeros(100)
            tau, ess = np.nan, np.nan
        
        elapsed = time.time() - start_time
        
        # Store results
        results[name] = {
            "samples": flat_samples,
            "acceptance_rates": acceptance_rates,
            "mean_radius": mean_radius,
            "radius_std": radius_std,
            "mean_distance_from_ring": mean_distance_from_ring,
            "autocorrelation": acf,
            "tau": tau,
            "ess": ess,
            "time": elapsed
        }
        
        print(f"  Acceptance rate: {np.mean(acceptance_rates):.2f}")
        print(f"  Mean radius: {mean_radius:.4f} ")
        print(f"  Radius std: {radius_std:.4f}")
        # print(f"  Mean distance from ring: {mean_distance_from_ring:.4f}")
        print(f"  Integrated autocorrelation time: {tau:.2f}" if np.isfinite(tau) else "  Integrated autocorrelation time: NaN")
        # print(f"  Effective sample size: {ess:.2f}" if np.isfinite(ess) else "  Effective sample size: NaN")
        # print(f"  ESS/sec: {ess/elapsed:.2f}" if np.isfinite(ess) else "  ESS/sec: NaN")
        print(f"  Time: {elapsed:.2f} seconds")
    
    return results, sigma

def plot_ring_results(results, dim=10, sigma=0.1):
    """Plot comparison of sampler results for ring distribution"""
    samplers = list(results.keys())
    
    # 1. Plot autocorrelation functions
    plt.figure(figsize=(12, 6))
    for i, name in enumerate(samplers):
        acf = results[name]["autocorrelation"]
        max_lag = min(300, len(acf))
        plt.plot(np.arange(max_lag), acf[:max_lag], label=name)
    
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation Functions (First Dimension)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ring_autocorrelation.png")
    
    # 2. Plot 2D projection of samples
    if dim >= 2:
        plt.figure(figsize=(15, 12))
        
        # Plot the ring as a reference
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        
        # Plot multiple rings showing the width of the distribution
        widths = [1, 2, 3]  # Standard deviations
        for w in widths:
            inner_r = 1 - w * sigma
            outer_r = 1 + w * sigma
            if inner_r > 0:  # Only plot inner circle if radius is positive
                plt.plot(inner_r * circle_x, inner_r * circle_y, 'k:', alpha=0.3)
            plt.plot(outer_r * circle_x, outer_r * circle_y, 'k:', alpha=0.3)
        
        # Plot unit circle (where the ring is centered)
        plt.plot(circle_x, circle_y, 'k--', alpha=0.5, label='Target Ring (r=1)')
        
        # Plot samples from each sampler
        colors = plt.cm.tab10(np.linspace(0, 1, len(samplers)))
        
        for i, name in enumerate(samplers):
            samples = results[name]["samples"]
            
            # Subsample for clarity
            if len(samples) > 1000:
                idx = np.random.choice(len(samples), 1000, replace=False)
                plot_samples = samples[idx]
            else:
                plot_samples = samples
            
            plt.scatter(plot_samples[:, 0], plot_samples[:, 1], 
                       color=colors[i], alpha=0.5, label=name)
        
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.title(f"Ring Distribution: First 2 Dimensions (σ = {sigma})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')  # Equal aspect ratio
        plt.tight_layout()
        plt.savefig("ring_samples_2d.png")

    return

# Run the benchmark for the ring distribution
# Note: You need to have the sampler functions (side_move, stretch_move, etc.) defined elsewhere
results, sigma = benchmark_samplers_ring(dim=50, n_samples=100000, burn_in=20000, sigma=0.5)

# Plot the results
plot_ring_results(results, dim=50, sigma=0.5)

print("Ring distribution benchmark complete. Check the output directory for plots.")