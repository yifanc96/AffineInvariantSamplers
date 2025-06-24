import numpy as np
import matplotlib.pyplot as plt
import time
import os
import wandb

from samplers import side_move, stretch_move, hmc, hamiltonian_walk_move, hamiltonian_side_move
from autocorrelation_func import autocorrelation_fft, integrated_autocorr_time


def benchmark_samplers_ring(dim=10, n_samples=10000, burn_in=1000, sigma=0.1, n_thin = 1, save_dir = None):
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
        "Side Move": lambda: side_move(log_density, initial, total_samples, n_walkers=dim*2, gamma=1.687, n_thin=n_thin),
        "Stretch Move": lambda: stretch_move(log_density, initial, total_samples, n_walkers=dim*2, a=1.0+2.151/np.sqrt(dim), n_thin=n_thin),
        "HMC n=10": lambda: hmc(log_density, initial, total_samples, gradient, epsilon=0.1, L=10, n_chains=1, n_thin=n_thin),
        "HMC n=2": lambda: hmc(log_density, initial, total_samples, gradient, epsilon=0.5, L=2, n_chains=1, n_thin=n_thin),
        "Hamiltonian Walk Move n=10": lambda: hamiltonian_walk_move(gradient, potential, initial, total_samples, 
                                                        n_chains_per_group=dim, epsilon=0.1, n_leapfrog=10, beta=1.0, n_thin=n_thin),
        "Hamiltonian Walk Move n=2": lambda: hamiltonian_walk_move(gradient, potential, initial, total_samples, 
                                                        n_chains_per_group=dim, epsilon=0.5, n_leapfrog=2, beta=1.0, n_thin=n_thin),
        "Hamitonian Side Move n=10": lambda: hamiltonian_side_move(gradient, potential, initial, total_samples,
                                                        n_chains_per_group=dim, epsilon=0.1, n_leapfrog=10, beta=1.0, n_thin=n_thin),
        "Hamitonian Side Move n=2": lambda: hamiltonian_side_move(gradient, potential, initial, total_samples, 
                                                                              n_chains_per_group=dim, epsilon=0.5, n_leapfrog=2, beta=1.0, n_thin=n_thin),
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
        
        if save_dir:
            np.save(os.path.join(save_dir, f"samples_{name}.npy"), post_burn_in_samples)
            np.save(os.path.join(save_dir, f"acf_{name}.npy"), acf)
    
    return results, sigma

n_samples = 10**5
burn_in = 2*10**4
dim = 50
n_thin = 10

home = "/scratch/yc3400/AffineInvariant/"
timestamp = time.strftime("%Y%m%d-%H%M%S")
folder = f"benchmark_results_ring_sample10_5_n_thin_10_{timestamp}"

wandb_project = "AffineInvariant"
wandb_entity = 'yifanc96'
wandb_run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        resume=None,
        id    =None,
        name = folder
        )
wandb.run.log_code(".")

print(f'n_sample{n_samples}, burn_in{burn_in}, n_thin{n_thin}')
print(f"dim={dim}")
save_dir = os.path.join(home+folder, f"{dim}")
    
if save_dir is not None:
    os.makedirs(save_dir, exist_ok=True)
    
# Run the benchmark for the ring distribution
# Note: You need to have the sampler functions (side_move, stretch_move, etc.) defined elsewhere
results, sigma = benchmark_samplers_ring(dim=dim, n_samples=n_samples, burn_in=burn_in, sigma=0.5, n_thin=n_thin, save_dir=save_dir)

