import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.stats import norm

def walk_move_k_subset_vectorized(log_prob, initial, n_samples, n_walkers=20, n_thin=1, subset_size=None, stepsize=1.0,
                                   target_accept=0.5, n_warmup=1000, gamma=0.05, t0=10, kappa=0.75):
    """
    Vectorized implementation of the Ensemble MCMC with walk moves.
    Each active walker gets its own unique, randomly selected k-subset from the complementary ensemble.
    
    Parameters:
    -----------
    log_prob : function
        Log probability density function that accepts array of shape (n_walkers, dim)
        and returns array of shape (n_walkers,)
    initial : array
        Initial state (will be used as mean for initializing walkers)
    n_samples : int
        Number of samples to draw per walker
    n_walkers : int
        Number of walkers in the ensemble (must be even)
    n_thin : int
        Thinning factor - store every n_thin sample (default: 1, no thinning)
    subset_size : int, optional
        Size of subset S from complementary ensemble to use for proposals.
        Must be >= 2. If None, uses all complementary walkers.
    stepsize : float
        Scale factor for the proposal step size (default: 1.0)
    target_accept : float
        Target acceptance rate for dual averaging (default: 0.5)
    n_warmup : int
        Number of warmup iterations for step size adaptation (default: 1000)
    gamma : float
        Dual averaging parameter controlling adaptation rate (default: 0.05)
    t0 : float
        Dual averaging parameter for numerical stability (default: 10)
    kappa : float
        Dual averaging parameter controlling decay (default: 0.75, should be in (0.5, 1])
        
    Returns:
    --------
    samples : array
        Samples from all walkers (shape: n_walkers, n_samples, dim)
    acceptance_rates : array
        Acceptance rates for all walkers
    """
    
    # Ensure even number of walkers
    if n_walkers % 2 != 0:
        n_walkers += 1
        
    dim = len(initial)
    half_walkers = n_walkers // 2
    
    # Set subset size for complementary ensemble
    if subset_size is None:
        subset_size = half_walkers
    else:
        subset_size = min(max(subset_size, 2), half_walkers)  # Ensure >= 2 and <= half_walkers
    
    # Initialize walkers with small random perturbations around initial
    walkers = np.tile(initial, (n_walkers, 1)) + 0.1 * np.random.randn(n_walkers, dim)
    
    # Vectorized evaluation of initial log probabilities
    walker_log_probs = log_prob(walkers)
    
    # Dual averaging initialization
    log_stepsize = np.log(stepsize)
    log_stepsize_bar = 0.0
    h_bar = 0.0 # Same variable name as in the EKM sampler
    
    # Calculate total iterations needed based on thinning factor
    total_sampling_iterations = n_samples * n_thin
    total_iterations = n_warmup + total_sampling_iterations
    
    # Storage for samples and tracking acceptance
    samples = np.zeros((n_walkers, n_samples, dim))
    accepts_warmup = np.zeros(n_walkers)
    accepts_sampling = np.zeros(n_walkers)
    
    # Sample index to track where to store thinned samples
    sample_idx = 0
    
    # Main sampling loop
    for i in range(total_iterations):
        is_warmup = i < n_warmup
        current_stepsize = stepsize if not is_warmup else np.exp(log_stepsize)
        
        # Store current state from all walkers (only every n_thin iterations)
        if not is_warmup and (i - n_warmup) % n_thin == 0 and sample_idx < n_samples:
            samples[:, sample_idx] = walkers
            sample_idx += 1
        
        # Update first half, then second half
        for half in [0, 1]:
            # Set indices for active and complementary walker sets
            active_indices = np.arange(half * half_walkers, (half + 1) * half_walkers)
            comp_indices = np.arange((1 - half) * half_walkers, (2 - half) * half_walkers)
            
            # --- Key Modification for unique k-subsets for each walker ---
            # 1. Generate random indices for each active walker
            selected_comp_indices = np.array([np.random.choice(comp_indices, size=subset_size, replace=False) 
                                              for _ in range(half_walkers)])
            
            # 2. Use advanced indexing to get the correct walkers and their mean
            selected_walkers = walkers[selected_comp_indices] # Shape: (half_walkers, subset_size, dim)
            comp_means = np.mean(selected_walkers, axis=1) # Shape: (half_walkers, dim)
            
            # 3. Compute centered walkers relative to their *own* selected subset mean
            centered_walkers = selected_walkers - np.expand_dims(comp_means, axis=1) # Shape: (half_walkers, subset_size, dim)
            
            # Extract active walkers
            active_walkers = walkers[active_indices]
            
            # 4. Generate all Z_j values at once for each active walker
            z_values = np.random.randn(half_walkers, subset_size)
            
            # 5. Use einsum for vectorized calculation of proposal offsets
            # The einsum operation handles the dot product for each of the half_walkers
            proposal_offsets = current_stepsize * np.einsum('ws,wsd->wd', z_values, centered_walkers)
            
            # Create proposals: X_k(t) -> X_k(t) + stepsize * W
            proposals = active_walkers + proposal_offsets
            
            # Evaluate all proposals at once
            proposal_log_probs = log_prob(proposals)
            
            # For walk moves, the acceptance probability is just the ratio of likelihoods
            # (symmetric proposal distribution)
            log_accept_probs = proposal_log_probs - walker_log_probs[active_indices]
            
            # Generate random numbers for acceptance decisions
            random_uniforms = np.log(np.random.uniform(size=half_walkers))
            
            # Determine which proposals are accepted
            accepted = random_uniforms < log_accept_probs
            
            # Update walkers and log probabilities in one step
            walkers[active_indices[accepted]] = proposals[accepted]
            walker_log_probs[active_indices[accepted]] = proposal_log_probs[accepted]
            
            # Track acceptance
            if is_warmup:
                accepts_warmup[active_indices[accepted]] += 1
            else:
                accepts_sampling[active_indices[accepted]] += 1

        # Dual averaging step size adaptation during warmup
        if is_warmup:
            # Average acceptance probability across all walkers in this iteration
            current_accept_rate = (np.sum(accepts_warmup) / (i + 1)) / n_walkers
            
            # Corrected dual averaging update (from EKM sampler)
            m = i + 1  # iteration number (1-indexed)
            eta_m = 1.0 / (m + t0)
            
            # Update log step size
            h_bar = (1 - eta_m) * h_bar + eta_m * (target_accept - current_accept_rate)
            
            # Compute log step size with shrinkage
            log_stepsize = np.log(stepsize) - np.sqrt(m) / gamma * h_bar
            
            # Update log_stepsize_bar for final step size
            eta_bar_m = m**(-kappa)
            log_stepsize_bar = (1 - eta_bar_m) * log_stepsize_bar + eta_bar_m * log_stepsize
        
        # After warmup, fix step size to the adapted value
        if i == n_warmup - 1:
            stepsize = np.exp(log_stepsize_bar)
            print(f"Warmup complete. Final adapted step size: {stepsize:.6f}")
    
    # Return results from all walkers
    acceptance_rates = accepts_sampling / total_sampling_iterations
    return samples, acceptance_rates

def autocorrelation_fft(x, max_lag=None):
    """
    Efficiently compute autocorrelation function using FFT.
    """
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 3, 20000)
    
    x_norm = x - np.mean(x)
    var = np.var(x_norm)
    x_norm = x_norm / np.sqrt(var)
    
    fft = np.fft.fft(x_norm, n=2*n)
    acf = np.fft.ifft(fft * np.conjugate(fft))[:n]
    acf = acf.real / n
    
    return acf[:max_lag]

def integrated_autocorr_time(x, M=5, c=10):
    """
    Estimate the integrated autocorrelation time using a self-consistent window.
    Based on the algorithm described by Goodman and Weare.
    """
    n = len(x)
    orig_x = x.copy()
    
    k_reductions = 0
    max_iterations = 10
    
    while k_reductions < max_iterations:
        acf = autocorrelation_fft(x)
        
        tau = 1.0
        for window in range(1, len(acf)):
            tau_window = 1.0 + 2.0 * sum(acf[1:window+1])
            if window <= M * tau_window:
                tau = tau_window
            else:
                break
        
        if n >= c * tau:
            tau = tau * (2**k_reductions)
            break
            
        k_reductions += 1
        n_half = len(x) // 2
        x_new = np.zeros(n_half)
        for i in range(n_half):
            if 2*i + 1 < len(x):
                x_new[i] = 0.5 * (x[2*i] + x[2*i+1])
            else:
                x_new[i] = x[2*i]
        x = x_new
        n = len(x)
    
    if k_reductions >= max_iterations or n < c * tau:
        acf = autocorrelation_fft(orig_x)
        tau = 1.0 + 2.0 * sum(acf[1:min(len(acf), int(M)+1)])
        tau = tau * (2**k_reductions)
    
    ess = len(orig_x) / tau
    
    return tau, acf, ess

def create_high_dim_precision(dim, condition_number=100):
    """Create a high-dimensional diagonal precision matrix with given condition number."""
    np.random.seed(42)
    eigenvalues = 0.1 * np.linspace(1, condition_number, dim)
    return eigenvalues

def benchmark_sampler_gaussian(dim, n_samples, n_warmup, n_chains_per_group, k, condition_number):
    """
    Benchmark a single sampler configuration on a high-dimensional Gaussian.
    """
    precision_diag = create_high_dim_precision(dim, condition_number)
    cov_diag = 1.0 / precision_diag
    true_mean = np.zeros(dim)

    def log_density(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        centered = x - true_mean
        result = -0.5 * np.sum(centered**2 * precision_diag, axis=1)
        return result

    initial = np.zeros(dim)
    n_walkers = 2 * n_chains_per_group
    
    start_time = time.time()
    samples, accept_rates = walk_move_k_subset_vectorized(
        log_prob=log_density,
        initial=initial,
        n_samples=n_samples,
        n_walkers=n_walkers,
        n_warmup=n_warmup,
        subset_size=k,
        stepsize=0.1,  # A conservative initial step size
        target_accept=0.45,
        n_thin=1
    )
    elapsed = time.time() - start_time
    
    flat_samples = samples.reshape(-1, dim)
    sample_mean = np.mean(flat_samples, axis=0)
    sample_var = np.var(flat_samples, axis=0)
    
    mean_mse = np.mean((sample_mean - true_mean)**2) / np.mean(true_mean**2) if np.mean(true_mean**2) != 0 else np.mean((sample_mean)**2)
    cov_mse = np.sum((sample_var - cov_diag)**2) / np.sum(cov_diag**2)
    
    v_samples_mean = np.mean(samples[:, :, 0], axis=0)
    tau, _, ess = integrated_autocorr_time(v_samples_mean)
    
    print(f"  k={k}, Mean Accept Rate: {np.mean(accept_rates):.3f}")
    print(f"  Mean MSE: {mean_mse:.6f}")
    print(f"  Covariance MSE: {cov_mse:.6f}")
    print(f"  Integrated Autocorrelation Time: {tau:.2f}")
    print(f"  Total Time: {elapsed:.2f}s")
    
    return samples, accept_rates

if __name__ == '__main__':
    dim_array = [4, 8, 16, 32]
    n_samples = 100000
    n_warmup = 10000
    condition_number = 1000
    k = None
    print(f"Benchmarking Walk Move Sampler on {dim_array}-D Gaussian...")


    for dim in dim_array:
        print(f"\nDimension: {dim}")
        n_chains_per_group = dim
        samples, accept_rates = benchmark_sampler_gaussian(
        dim=dim, n_samples=n_samples, n_warmup=n_warmup, 
        n_chains_per_group=n_chains_per_group, k=k, condition_number=condition_number
    )
    


