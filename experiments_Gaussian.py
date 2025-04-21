import numpy as np
import matplotlib.pyplot as plt
import time

from samplers import side_move, stretch_move, hmc, hamiltonian_walk_move, hamiltonian_side_move
from autocorrelation_func import autocorrelation_fft, integrated_autocorr_time

def create_high_dim_precision(dim, condition_number=100):
    """Create a high-dimensional precision matrix with given condition number."""
    # Create random eigenvectors (orthogonal matrix)
    np.random.seed(42)  # For reproducibility
    H = np.random.randn(dim, dim)
    Q, R = np.linalg.qr(H)
    
    # Create eigenvalues with desired condition number
    eigenvalues = 0.1* np.linspace(1, condition_number, dim)
    
    # Construct precision matrix: Q @ diag(eigenvalues) @ Q.T
    precision = Q @ np.diag(eigenvalues) @ Q.T
    
    # Ensure it's symmetric (fix numerical issues)
    precision = 0.5 * (precision + precision.T)
    
    return precision

def benchmark_samplers(dim=40, n_samples=10000, burn_in=1000, condition_number=100):
    """
    Benchmark different MCMC samplers on a high-dimensional Gaussian.
    """
    # Create precision matrix (inverse covariance)
    precision_matrix = create_high_dim_precision(dim, condition_number)
    
    # Compute covariance matrix for reference (needed for evaluation)
    cov_matrix = np.linalg.inv(precision_matrix)
    
    true_mean = np.ones(dim)
    

    def log_density(x):
        """Vectorized log density of the multivariate Gaussian"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # Vectorized operation for all samples
        centered = x - true_mean
        # Using einsum for efficient batch matrix multiplication
        result = -0.5 * np.einsum('ij,jk,ik->i', centered, precision_matrix, centered)
            
        return result
    
    def gradient(x):
        """Vectorized gradient of the negative log density (potential gradient)"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # Vectorized operation for all samples
        centered = x - true_mean
        # Matrix multiplication for each sample in the batch
        result = np.einsum('jk,ij->ik', precision_matrix, centered)
            
        return result
    
    def potential(x):
        """Vectorized negative log density (potential energy)"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # Vectorized operation for all samples
        centered = x - true_mean
        result = 0.5 * np.einsum('ij,jk,ik->i', centered, precision_matrix, centered)
            
        return result
    
    # Initial state
    initial = np.zeros(dim)
    
    # Dictionary to store results
    results = {}
    
    # Define samplers to benchmark with burn-in
    total_samples = n_samples + burn_in
    
    # Define samplers to benchmark - adjust parameters for high-dimensional case
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
        sample_cov = np.cov(flat_samples, rowvar=False)
        # Calculate mean squared error for mean and covariance
        mean_mse = np.mean((sample_mean - true_mean)**2) / np.mean(true_mean**2)
        cov_mse = np.sum((sample_cov - cov_matrix)**2) / np.sum(cov_matrix**2)
        
        # Compute autocorrelation for first dimension
        # Average over chains to compute autocorrelation
        acf = autocorrelation_fft(np.mean(samples[:, :, 0],axis=0))
        
        # Compute integrated autocorrelation time for first dimension
        try:
            tau, _, ess = integrated_autocorr_time(np.mean(samples[:, :, 0],axis=0))
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
    
    return results, true_mean, cov_matrix

def plot_results(results, dim=40, true_mean=None, cov_matrix=None, condition_number=100):
    """Plot comparison of sampler results"""
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
    plt.savefig("autocorrelation.png")
    
    return

# Main execution
# Run benchmarks with higher dimensionality, challenging covariance structure, and burn-in
results, true_mean, cov_matrix = benchmark_samplers(dim=50, n_samples=100000, burn_in=20000, condition_number=1000)

# Plot results
# plot_results(results, dim=20, true_mean=true_mean, cov_matrix=cov_matrix, condition_number=1000)

# You may need to install the corner package if not already available
# Uncomment the following to install if needed:
# import sys
# import subprocess
# subprocess.check_call([sys.executable, "-m", "pip", "install", "corner"])

print("Benchmark complete. Check the output directory for plots.")