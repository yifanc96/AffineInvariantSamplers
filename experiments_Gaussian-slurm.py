import numpy as np
import matplotlib.pyplot as plt
import time
import os
import wandb

from samplers import side_move, stretch_move, hmc, hamiltonian_walk_move, hamiltonian_side_move
from autocorrelation_func import autocorrelation_fft, integrated_autocorr_time

def create_high_dim_precision(dim, condition_number=100):
    """Create a high-dimensional diagonal precision matrix with given condition number."""
    # For reproducibility
    np.random.seed(42)
    
    # Create diagonal eigenvalues with desired condition number
    eigenvalues = 0.1 * np.linspace(1, condition_number, dim)
    
    # For diagonal matrices, we can just return the eigenvalues
    # This avoids storing the full matrix which is mostly zeros
    return eigenvalues

def benchmark_samplers(dim=40, n_samples=10000, burn_in=1000, condition_number=100, n_thin = 1, save_dir = None):
    """
    Benchmark different MCMC samplers on a high-dimensional Gaussian.
    """
    # Create precision matrix (inverse covariance) - just the diagonal values
    precision_diag = create_high_dim_precision(dim, condition_number)
    
    # Compute covariance matrix diagonal for reference (needed for evaluation)
    # For diagonal matrices, inverse is just reciprocal of diagonal elements
    cov_diag = 1.0 / precision_diag
    
    true_mean = np.ones(dim)
    
    def log_density(x):
        """Optimized log density of the multivariate Gaussian with diagonal precision"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # Vectorized operation for all samples using broadcasting
        centered = x - true_mean
        # For diagonal precision, this simplifies to sum of elementwise products
        # This avoids the expensive einsum operation
        result = -0.5 * np.sum(centered**2 * precision_diag, axis=1)
            
        return result
    
    def gradient(x):
        """Optimized gradient of the negative log density with diagonal precision"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # Vectorized operation for all samples
        centered = x - true_mean
        # For diagonal precision, this is just elementwise multiplication
        # This avoids the expensive einsum operation
        result = centered * precision_diag[np.newaxis, :]
            
        return result
    
    def potential(x):
        """Optimized negative log density (potential energy) with diagonal precision"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # Vectorized operation for all samples
        centered = x - true_mean
        # For diagonal precision, this simplifies to sum of elementwise products
        result = 0.5 * np.sum(centered**2 * precision_diag, axis=1)
            
        return result
    
    # Initial state
    initial = np.zeros(dim)
    
    # Dictionary to store results
    results = {}
    
    # Define samplers to benchmark with burn-in
    total_samples = n_samples + burn_in
    
    # Define samplers to benchmark - adjust parameters for high-dimensional case
    samplers = {
        "Side Move": lambda: side_move(log_density, initial, total_samples, n_walkers=dim*2, gamma=1.687, n_thin=n_thin),
        "Stretch Move": lambda: stretch_move(log_density, initial, total_samples, n_walkers=dim*2, a=1.0+2.151/np.sqrt(dim),n_thin=n_thin),
        "HMC n=10": lambda: hmc(log_density, initial, total_samples, gradient, epsilon=0.1, L=10, n_chains=1,n_thin=n_thin),
        "HMC n=2": lambda: hmc(log_density, initial, total_samples, gradient, epsilon=0.5, L=2, n_chains=1,n_thin=n_thin),
        "Hamiltonian Walk Move n=10": lambda: hamiltonian_walk_move(gradient, potential, initial, total_samples, 
                                                        n_chains_per_group=dim, epsilon=0.1, n_leapfrog=10, beta=1.0,n_thin=n_thin),
        "Hamiltonian Walk Move n=2": lambda: hamiltonian_walk_move(gradient, potential, initial, total_samples, 
                                                        n_chains_per_group=dim, epsilon=0.5, n_leapfrog=2, beta=1.0,n_thin=n_thin),
        "Hamiltonian Side Move n=10": lambda: hamiltonian_side_move(gradient, potential, initial, total_samples,
                                                        n_chains_per_group=dim, epsilon=0.1, n_leapfrog=10, beta=1.0,n_thin=n_thin),
        "Hamiltonian Side Move n=2": lambda: hamiltonian_side_move(gradient, potential, initial, total_samples, 
                                                                              n_chains_per_group=dim, epsilon=0.5, n_leapfrog=2, beta=1.0,n_thin=n_thin),
    }
    
    for name, sampler_func in samplers.items():
        print(f"Running {name}...")
        start_time = time.time()
        samples, acceptance_rates = sampler_func()
        elapsed = time.time() - start_time
        
        # Apply burn-in: discard the first burn_in samples
        post_burn_in_samples = samples[:, burn_in:, :]
        
        # Flatten samples from all chains
        flat_samples = post_burn_in_samples.reshape(-1, dim)
        
        # Compute sample mean and covariance
        sample_mean = np.mean(flat_samples, axis=0)
        
        # For MSE calculation, we don't need to compute the full covariance matrix
        # We can compute the diagonal elements directly
        sample_var = np.var(flat_samples, axis=0)
        
        # Calculate mean squared error for mean and covariance
        mean_mse = np.mean((sample_mean - true_mean)**2) / np.mean(true_mean**2)
        # For diagonal covariance, we only compare diagonal elements
        cov_mse = np.sum((sample_var - cov_diag)**2) / np.sum(cov_diag**2)
        
        # Compute autocorrelation for first dimension
        # Average over chains to compute autocorrelation
        acf = autocorrelation_fft(np.mean(post_burn_in_samples[:, :, 0], axis=0))
        
        # Compute integrated autocorrelation time for first dimension
        try:
            tau, _, ess = integrated_autocorr_time(np.mean(post_burn_in_samples[:, :, 0], axis=0))
        except:
            tau, ess = np.nan, np.nan
        
        # Store results
        results[name] = {
            "samples": flat_samples,
            "acceptance_rates": acceptance_rates,
            "mean_mse": mean_mse,
            "cov_mse": cov_mse,
            "autocorrelation": acf,
            "tau": tau,
            "ess": ess,
            "time": elapsed
        }
        
        print(f"  Acceptance rate: {np.mean(acceptance_rates):.2f}")
        print(f"  Mean MSE: {mean_mse:.6f}")
        print(f"  Covariance MSE: {cov_mse:.6f}")
        print(f"  Integrated autocorrelation time: {tau:.2f}")
        print(f"  Time: {elapsed:.2f} seconds")

        if save_dir:
            np.save(os.path.join(save_dir, f"samples_{name}.npy"), post_burn_in_samples)
            np.save(os.path.join(save_dir, f"acf_{name}.npy"), acf)
            
    return results, true_mean, cov_diag
    

n_samples = 10**5
burn_in = 2*10**4
array_dim = [4, 8, 16, 32, 64, 128]
n_thin = 10
# array_dim = [128]


home = "/scratch/yc3400/AffineInvariant/"
timestamp = time.strftime("%Y%m%d-%H%M%S")
folder = f"benchmark_results_Gaussian_sample10_5_n_thin_10_{timestamp}"

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
    
for dim in array_dim:
    print(f"dim={dim}")
    # Create a timestamped directory for this run
    save_dir = os.path.join(home+folder, f"{dim}")
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Run benchmarks and save results
    results, true_mean, cov_diag = benchmark_samplers(
    dim=dim, 
    n_samples=n_samples, 
    burn_in=burn_in, 
    condition_number=1000,
    n_thin=n_thin,
    save_dir=save_dir
)
