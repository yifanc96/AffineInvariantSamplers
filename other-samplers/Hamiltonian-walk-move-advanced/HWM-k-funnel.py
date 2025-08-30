import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.stats import norm


def hamiltonian_walk_move_k_subset(gradient_func, potential_func, initial, n_samples, k=None,
                                             n_chains_per_group=5, epsilon_init=0.01, n_leapfrog=10, 
                                             beta=0.05, n_thin=1, target_accept=0.65, n_warmup=1000,
                                             gamma=0.05, t0=10, kappa=0.75):
    """
    Vectorized Hamiltonian Walk Move sampler using k-subset of complementary ensemble for preconditioning.
    
    This implementation ensures each chain uses a different, randomly sampled k-subset
    from the complementary ensemble for its leapfrog integration.
    
    Parameters:
    -----------
    gradient_func : callable
        Function that computes gradients of the negative log probability (potential energy)
    potential_func : callable  
        Function that computes the negative log probability (potential energy)
    initial : array_like
        Initial state
    n_samples : int
        Number of samples to collect (after warmup)
    k : int, optional
        Number of chains to use from complementary ensemble (default: n_chains_per_group)
        Must satisfy 2 <= k <= n_chains_per_group
    n_chains_per_group : int
        Number of chains per group (default: 5)
    epsilon_init : float
        Initial step size (default: 0.01)
    n_leapfrog : int
        Number of leapfrog steps (default: 10)
    beta : float
        Preconditioning parameter (default: 0.05)
    n_thin : int
        Thinning factor - store every n_thin sample (default: 1, no thinning)
    target_accept : float
        Target acceptance rate for dual averaging (default: 0.65)
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
    samples : ndarray
        Generated samples from all chains (after warmup)
    acceptance_rates : ndarray
        Final acceptance rates for each chain
    step_size_history : ndarray
        History of step sizes during adaptation
    """
    
    # Initialize
    orig_dim = initial.shape
    flat_dim = np.prod(orig_dim)
    total_chains = 2 * n_chains_per_group
    
    # Set default k
    if k is None:
        k = n_chains_per_group
    
    # Validate k
    if k < 2 or k > n_chains_per_group:
        raise ValueError(f"k must satisfy 2 <= k <= n_chains_per_group, got k={k}")
    
    # Create initial states with small random perturbations
    states = np.tile(initial.flatten(), (total_chains, 1)) + 0.1 * np.random.randn(total_chains, flat_dim)
    
    # Split into two groups
    group1_idx = slice(0, n_chains_per_group)
    group2_idx = slice(n_chains_per_group, total_chains)
    
    # Dual averaging initialization
    h = epsilon_init
    log_h = np.log(h)
    log_h_bar = 0.0
    h_bar = 0.0  # FIX: Initialize h_bar for dual averaging
    step_size_history = []
    
    # Calculate total iterations needed based on thinning factor
    total_sampling_iterations = n_samples * n_thin
    total_iterations = n_warmup + total_sampling_iterations
    
    # Storage for samples and acceptance tracking
    samples = np.zeros((total_chains, n_samples, flat_dim))
    accepts_warmup = np.zeros(total_chains)  # Track accepts during warmup
    accepts_sampling = np.zeros(total_chains)  # Track accepts during sampling
    
    # Sample index to track where to store thinned samples
    sample_idx = 0
    
    print(f"Using a unique random k-subset for each chain (k={k})")
    
    # Main sampling loop
    for i in range(total_iterations):
        is_warmup = i < n_warmup
        current_epsilon = epsilon_init if not is_warmup else np.exp(log_h)
        
        # Store current state from all chains (only during sampling phase)
        if not is_warmup and (i - n_warmup) % n_thin == 0 and sample_idx < n_samples:
            samples[:, sample_idx] = states
            sample_idx += 1
        
        # Precompute step size terms
        beta_eps = beta * current_epsilon
        beta_eps_half = beta_eps / 2
        
        # --- Vectorized Update for Group 1 ---
        # Each chain in group 1 uses a unique random k-subset from group 2
        
        # 1. Generate random k-subsets for each chain in group 1 from group 2 chains
        selected_indices_2 = np.array([np.random.choice(n_chains_per_group, size=k, replace=False)
                                       for _ in range(n_chains_per_group)]) # (n_cpg, k)
        
        # 2. Use advanced indexing to get all needed states
        selected_states_2 = states[group2_idx][selected_indices_2] # (n_cpg, k, flat_dim)
        
        # 3. Centering and normalization for each k-subset using broadcasting
        centered2 = (selected_states_2 - np.mean(selected_states_2, axis=1, keepdims=True)) / np.sqrt(k)
        
        # Store current state and energy
        current_q1 = states[group1_idx].copy()
        current_q1_reshaped = current_q1.reshape(n_chains_per_group, *orig_dim)
        current_U1 = potential_func(current_q1_reshaped)
        
        # Initialize momentum in k-dimensional subspace
        p1 = np.random.randn(n_chains_per_group, k)
        current_K1 = np.clip(0.5 * np.sum(p1**2, axis=1), 0, 1000)
        
        # Leapfrog integration
        q1 = current_q1.copy()
        p1_current = p1.copy()
        
        # Initial half-step for momentum
        grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim))
        grad1 = np.nan_to_num(grad1, nan=0.0)
        p1_current -= beta_eps_half * np.einsum('np,nsp->ns', grad1, centered2)

        for step in range(n_leapfrog):
            # Position update with ensemble preconditioning
            q1 += beta_eps * np.einsum('ns,nsp->np', p1_current, centered2)

            if step < n_leapfrog - 1:
                # Momentum update
                grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim))
                grad1 = np.nan_to_num(grad1, nan=0.0)
                p1_current -= beta_eps * np.einsum('np,nsp->ns', grad1, centered2)
        
        # Final half-step for momentum
        grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim))
        grad1 = np.nan_to_num(grad1, nan=0.0)
        p1_current -= beta_eps_half * np.einsum('np,nsp->ns', grad1, centered2)
        
        # Compute proposed energy
        proposed_U1 = potential_func(q1.reshape(n_chains_per_group, *orig_dim))
        proposed_K1 = np.clip(0.5 * np.sum(p1_current**2, axis=1), 0, 1000)
        
        # Metropolis acceptance with numerical stability
        dH1 = (proposed_U1 + proposed_K1) - (current_U1 + current_K1)
        
        accept_probs1 = np.ones_like(dH1)
        exp_needed = dH1 > 0
        if np.any(exp_needed):
            safe_dH = np.clip(dH1[exp_needed], None, 100)
            accept_probs1[exp_needed] = np.exp(-safe_dH)
        
        accepts1 = np.random.random(n_chains_per_group) < accept_probs1
        states[group1_idx][accepts1] = q1[accepts1]
        
        # Track acceptances
        if is_warmup:
            accepts_warmup[group1_idx] += accepts1
        else:
            accepts_sampling[group1_idx] += accepts1
        
        # --- Vectorized Update for Group 2 (Symmetric) ---
        # Each chain in group 2 uses a unique random k-subset from group 1
        
        selected_indices_1 = np.array([np.random.choice(n_chains_per_group, size=k, replace=False)
                                       for _ in range(n_chains_per_group)]) # (n_cpg, k)
        
        selected_states_1 = states[group1_idx][selected_indices_1]
        
        centered1 = (selected_states_1 - np.mean(selected_states_1, axis=1, keepdims=True)) / np.sqrt(k)
        
        current_q2 = states[group2_idx].copy()
        current_q2_reshaped = current_q2.reshape(n_chains_per_group, *orig_dim)
        current_U2 = potential_func(current_q2_reshaped)
        
        p2 = np.random.randn(n_chains_per_group, k)
        current_K2 = np.clip(0.5 * np.sum(p2**2, axis=1), 0, 1000)
        
        q2 = current_q2.copy()
        p2_current = p2.copy()
        
        grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim))
        grad2 = np.nan_to_num(grad2, nan=0.0)
        p2_current -= beta_eps_half * np.einsum('np,nsp->ns', grad2, centered1)

        for step in range(n_leapfrog):
            q2 += beta_eps * np.einsum('ns,nsp->np', p2_current, centered1)
            
            if step < n_leapfrog - 1:
                grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim))
                grad2 = np.nan_to_num(grad2, nan=0.0)
                p2_current -= beta_eps * np.einsum('np,nsp->ns', grad2, centered1)
        
        grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim))
        grad2 = np.nan_to_num(grad2, nan=0.0)
        p2_current -= beta_eps_half * np.einsum('np,nsp->ns', grad2, centered1)
        
        proposed_U2 = potential_func(q2.reshape(n_chains_per_group, *orig_dim))
        proposed_K2 = np.clip(0.5 * np.sum(p2_current**2, axis=1), 0, 1000)
        
        dH2 = (proposed_U2 + proposed_K2) - (current_U2 + current_K2)
        
        accept_probs2 = np.ones_like(dH2)
        exp_needed = dH2 > 0
        if np.any(exp_needed):
            safe_dH = np.clip(dH2[exp_needed], None, 100)
            accept_probs2[exp_needed] = np.exp(-safe_dH)
            
        accepts2 = np.random.random(n_chains_per_group) < accept_probs2
        states[group2_idx][accepts2] = q2[accepts2]
        
        if is_warmup:
            accepts_warmup[group2_idx] += accepts2
        else:
            accepts_sampling[group2_idx] += accepts2
        
        # Dual averaging step size adaptation during warmup
        if is_warmup:
            # Average acceptance probability across all chains in this iteration
            current_accept_rate = (np.sum(accepts1) + np.sum(accepts2)) / total_chains
            
            # Dual averaging update
            m = i + 1  # iteration number (1-indexed)
            eta_m = 1.0 / (m + t0)
            
            # Update log step size
            h_bar = (1 - eta_m) * h_bar + eta_m * (target_accept - current_accept_rate)
            
            # Compute log step size with shrinkage
            log_h = np.log(h) - np.sqrt(m) / gamma * h_bar
            
            # Update log_h_bar for final step size
            eta_bar_m = m**(-kappa)
            log_h_bar = (1 - eta_bar_m) * log_h_bar + eta_bar_m * log_h
            
            # Store step size history
            step_size_history.append(np.exp(log_h))
        
        # After warmup, fix step size to the adapted value
        if i == n_warmup - 1:
            epsilon_init = np.exp(log_h)
            print(f"Warmup complete. Final adapted step size: {epsilon_init:.6f}")
    
    # Reshape final samples to original dimensions
    samples = samples.reshape((total_chains, n_samples) + orig_dim)
    
    # Compute acceptance rates for sampling phase only
    acceptance_rates = accepts_sampling / total_sampling_iterations
    
    return samples, acceptance_rates, np.array(step_size_history)

def autocorrelation_fft(x, max_lag=None):
    """
    Efficiently compute autocorrelation function using FFT.
    
    Parameters:
    -----------
    x : array
        1D array of samples
    max_lag : int, optional
        Maximum lag to compute (default: len(x)//3)
        
    Returns:
    --------
    acf : array
        Autocorrelation function values
    """
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 3, 20000)  # Cap at 20000 to prevent slow computation
    
    # Remove mean and normalize
    x_norm = x - np.mean(x)
    var = np.var(x_norm)
    if var == 0:
        return np.zeros(max_lag)
    x_norm = x_norm / np.sqrt(var)
    
    # Compute autocorrelation using FFT
    # Pad the signal with zeros to avoid circular correlation
    fft = np.fft.fft(x_norm, n=2*n)
    acf = np.fft.ifft(fft * np.conjugate(fft))[:n]
    acf = acf.real / n  # Normalize
    
    return acf[:max_lag]

def integrated_autocorr_time(x, M=5, c=10):
    """
    Estimate the integrated autocorrelation time using a self-consistent window.
    Based on the algorithm described by Goodman and Weare.
    
    Parameters:
    -----------
    x : array
        1D array of samples
    M : int, default=5
        Window size multiplier (typically 5-10)
    c : int, default=10
        Maximum lag cutoff for window determination
        
    Returns:
    --------
    tau : float
        Integrated autocorrelation time
    acf : array
        Autocorrelation function values
    ess : float
        Effective sample size
    """
    n = len(x)
    orig_x = x.copy()
    
    # Initial pairwise reduction if needed
    k = 0
    max_iterations = 10  # Prevent infinite loop
    
    while k < max_iterations:
        # Calculate autocorrelation function
        acf = autocorrelation_fft(x)
        
        # Calculate integrated autocorrelation time with self-consistent window
        tau = 1.0  # Initialize with the first term
        
        # Find the window size where window <= M * tau
        for window in range(1, len(acf)):
            # Update tau with this window
            tau_window = 1.0 + 2.0 * sum(acf[1:window+1])
            
            # Check window consistency: window <= M*tau
            if window <= M * tau_window:
                tau = tau_window
            else:
                break
        
        # If we have a robust estimate, we're done
        if n >= c * tau:
            # Scale tau back to the original time scale: tau_0 = 2^k * tau_k
            tau = tau * (2**k)
            break
            
        # If we don't have a robust estimate, perform pairwise reduction
        k += 1
        n_half = len(x) // 2
        x_new = np.zeros(n_half)
        for i in range(n_half):
            if 2*i + 1 < len(x):
                x_new[i] = 0.5 * (x[2*i] + x[2*i+1])
            else:
                x_new[i] = x[2*i]
        x = x_new
        n = len(x)
    
    # If we exited without a robust estimate, compute one final estimate
    if k >= max_iterations or n < c * tau:
        acf = autocorrelation_fft(orig_x)
        tau_reduced = 1.0 + 2.0 * sum(acf[1:min(len(acf), int(M)+1)])
        # Scale tau back to the original time scale
        tau = tau_reduced * (2**k)
    
    # Calculate effective sample size using original series length
    ess = len(orig_x) / tau
    
    return tau, acf, ess

def funnel_log_density(x, sigma_theta=3.0):
    """
    Log density of Neal's funnel distribution with numerical stability.
    
    Neal's funnel distribution is defined as:
    - theta ~ Normal(0, sigma_theta^2)  [Neal uses sigma_theta = 3]
    - x_i | theta ~ Normal(0, exp(theta/2)^2) = Normal(0, exp(theta)) for i = 1, ..., d-1
    
    Parameters:
    -----------
    x : array_like
        Input array of shape (..., d) where d is the dimension
    sigma_theta : float
        Standard deviation for the theta parameter (default: 3.0)
        
    Returns:
    --------
    log_prob : array
        Log probability density values
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    dim = x.shape[-1]
    theta = np.clip(x[..., 0], -10, 10)  # Clip theta to prevent overflow
    x_rest = x[..., 1:]  # Remaining dimensions
    
    # Log density components
    # p(theta) = Normal(0, sigma_theta^2)
    log_p_theta = -0.5 * (theta / sigma_theta)**2 - 0.5 * np.log(2 * np.pi * sigma_theta**2)
    
    # p(x_i | theta) = Normal(0, exp(theta/2)^2) = Normal(0, exp(theta))
    # log p(x_i | theta) = -0.5 * x_i^2 / exp(theta) - 0.5 * log(2*pi) - theta/2
    log_p_x_given_theta = -0.5 * np.sum(x_rest**2, axis=-1) * np.exp(-theta)
    log_p_x_given_theta -= 0.5 * (dim - 1) * np.log(2 * np.pi)
    log_p_x_given_theta -= (dim - 1) * theta / 2  # This is the correct term!
    
    # Handle extreme theta values
    result = log_p_theta + log_p_x_given_theta
    result = np.where(np.isfinite(result), result, -np.inf)
    
    return result

def funnel_gradient(x, sigma_theta=3.0):
    """
    Gradient of the negative log density (for HMC) with numerical stability.
    Corrected for Neal's funnel: x_i | theta ~ Normal(0, exp(theta/2)^2)
    
    Parameters:
    -----------
    x : array_like
        Input array of shape (..., d)
    sigma_theta : float
        Standard deviation for the theta parameter
        
    Returns:
    --------
    grad : array
        Gradient of negative log density
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    dim = x.shape[-1]
    theta = np.clip(x[..., 0], -10, 10)  # Clip theta to prevent overflow
    x_rest = x[..., 1:]
    
    # Gradient w.r.t. theta
    grad_theta = theta / (sigma_theta**2)  # From p(theta)
    
    # From p(x|theta) - corrected for Neal's funnel
    exp_neg_theta = np.exp(-theta)
    exp_neg_theta = np.clip(exp_neg_theta, 0, 1e10)  # Prevent extreme values
    
    # Gradient from the quadratic term: d/d_theta[-0.5 * sum(x_i^2) * exp(-theta)]
    grad_theta += 0.5 * np.sum(x_rest**2, axis=-1) * exp_neg_theta
    
    # Gradient from the normalization term: d/d_theta[-(dim-1) * theta/2]
    grad_theta += (dim - 1) / 2
    
    # Gradient w.r.t. x_rest: d/d_x_i[-0.5 * x_i^2 * exp(-theta)]
    grad_x_rest = x_rest * exp_neg_theta[..., np.newaxis]
    
    # Combine gradients
    grad = np.concatenate([grad_theta[..., np.newaxis], grad_x_rest], axis=-1)
    
    # Handle non-finite values
    grad = np.where(np.isfinite(grad), grad, 0.0)
    
    return grad

def funnel_potential(x, sigma_theta=3.0):
    """
    Potential energy (negative log density) for the funnel distribution with numerical stability.
    
    Parameters:
    -----------
    x : array_like
        Input array of shape (..., d)
    sigma_theta : float
        Standard deviation for the theta parameter
        
    Returns:
    --------
    potential : array
        Potential energy values
    """
    log_dens = funnel_log_density(x, sigma_theta)
    potential = -log_dens
    
    # Handle infinite log density (zero probability regions)
    potential = np.where(np.isfinite(potential), potential, 1e10)
    
    return potential

def compute_autocorrelation(chain, max_lag=50):
    """Compute autocorrelation function up to max_lag."""
    return autocorrelation_fft(chain, max_lag)

def effective_sample_size(chain, max_lag=None):
    """Estimate effective sample size using autocorrelation."""
    tau_int, acf, ess = integrated_autocorr_time(chain, M=5, c=10)
    return ess, tau_int

def test_funnel_k_values():
    """Test HWM with different k values on the funnel distribution."""
    
    # Problem setup
    dim = 5  # Total dimensions (v + 9 other dimensions)
    initial = np.zeros(dim)
    initial[0] = 0.0  # Start v near 0
    initial[1:] = 0.1 * np.random.randn(dim-1)
    
    n_samples = 400000
    n_warmup = 20000
    n_chains_per_group = 4*dim
    
    # Different k values to test
    # k_values = [2,n_chains_per_group]  # k=n_chains_per_group is full ensemble (original HWM)
    k_values = [2]
    results = {}
    
    print("Testing funnel distribution with different k values...")
    print(f"Dimension: {dim}, Chains per group: {n_chains_per_group}")
    print("-" * 60)
    
    for k in k_values:
        print(f"Running HWM with k={k}...")
        
        samples, accept_rates, step_history = hamiltonian_walk_move_k_subset(
            gradient_func=funnel_gradient, 
            potential_func=funnel_potential, 
            initial=initial, 
            n_samples=n_samples, 
            n_warmup=n_warmup, 
            n_chains_per_group=dim, 
            epsilon_init=1/(dim**(1/4)), 
            n_leapfrog=5, 
            beta=1.0,
            target_accept=0.2, 
            n_thin=1,
            k=2
        )

        
        # Extract v (first dimension) from all chains
        v_samples = samples[:, :, 0].flatten()  # Shape: (total_chains * n_samples,)
        
        # Compute statistics
        mean_accept = np.mean(accept_rates)
        final_step_size = step_history[-1] if len(step_history) > 0 else 0.1
        
        # Autocorrelation analysis for v
        ess_v, tau_int_v = effective_sample_size(v_samples)
        
        # Store results
        results[k] = {
            'samples': samples,
            'v_samples': v_samples,
            'accept_rate': mean_accept,
            'final_step_size': final_step_size,
            'ess_v': ess_v,
            'tau_int_v': tau_int_v,
            'v_mean': np.mean(v_samples),
            'v_std': np.std(v_samples)
        }
        
        print(f"  k={k}: Accept rate = {mean_accept:.3f}, ESS(v) = {ess_v:.1f}, τ_int(v) = {tau_int_v:.2f}")
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('HWM k-subset Analysis: Funnel Distribution', fontsize=16)
    
    # Plot 1: Acceptance rates vs k
    k_list = list(k_values)
    accept_rates_list = [results[k]['accept_rate'] for k in k_list]
    axes[0, 0].plot(k_list, accept_rates_list, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('k (subset size)')
    axes[0, 0].set_ylabel('Acceptance Rate')
    axes[0, 0].set_title('Acceptance Rate vs k')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Plot 2: Effective Sample Size vs k
    ess_list = [results[k]['ess_v'] for k in k_list]
    axes[0, 1].plot(k_list, ess_list, 'o-', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_xlabel('k (subset size)')
    axes[0, 1].set_ylabel('ESS for v')
    axes[0, 1].set_title('Effective Sample Size vs k')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Integrated Autocorrelation Time vs k
    tau_list = [results[k]['tau_int_v'] for k in k_list]
    axes[0, 2].plot(k_list, tau_list, 'o-', linewidth=2, markersize=8, color='red')
    axes[0, 2].set_xlabel('k (subset size)')
    axes[0, 2].set_ylabel('τ_int for v')
    axes[0, 2].set_title('Autocorrelation Time vs k')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Marginal distribution of v for different k values
    colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))
    for i, k in enumerate(k_values):
        v_samples = results[k]['v_samples']
        axes[1, 0].hist(v_samples, bins=50, alpha=0.6, density=True, 
                       color=colors[i], label=f'k={k}', histtype='stepfilled')
    
    # Theoretical distribution N(0, 3)
    x_theo = np.linspace(-10, 10, 100)
    y_theo = norm.pdf(x_theo, 0, 3)
    axes[1, 0].plot(x_theo, y_theo, 'k--', linewidth=2, label='True N(0,9)')
    axes[1, 0].set_xlabel('v (first dimension)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Marginal Distribution of v')
    axes[1, 0].legend()
    axes[1, 0].set_xlim(-10, 10)
    
    # Plot 5: Autocorrelation functions
    max_lag = 5000
    for i, k in enumerate(k_values):
        v_samples = results[k]['v_samples']
        autocorr = compute_autocorrelation(v_samples, max_lag)
        lags = np.arange(len(autocorr))
        axes[1, 1].plot(lags, autocorr, '-', color=colors[i], label=f'k={k}', linewidth=2)
    
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=0.05, color='red', linestyle=':', alpha=0.5, label='5% cutoff')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('Autocorrelation')
    axes[1, 1].set_title('Autocorrelation Functions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics table
    axes[1, 2].axis('off')
    
    # Create summary table
    table_data = []
    headers = ['k', 'Accept Rate', 'ESS(v)', 'τ_int(v)', 'Mean(v)', 'Std(v)']
    
    for k in k_values:
        row = [
            k,
            f"{results[k]['accept_rate']:.3f}",
            f"{results[k]['ess_v']:.1f}",
            f"{results[k]['tau_int_v']:.2f}",
            f"{results[k]['v_mean']:.3f}",
            f"{results[k]['v_std']:.3f}"
        ]
        table_data.append(row)
    
    # Add theoretical values
    table_data.append(['True', '-', '-', '-', '0.000', '3.000'])
    
    # Create table
    table = axes[1, 2].table(cellText=table_data, colLabels=headers,
                            cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 2].set_title('Summary Statistics')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed summary
    print("\n" + "="*60)
    print("DETAILED RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nTrue distribution: v ~ N(0, 3), x_i|v ~ N(0, exp(v/2))")
    print(f"Theoretical: Mean(v) = 0.000, Std(v) = 3.000")
    print()
    
    best_ess_k = max(k_values, key=lambda k: results[k]['ess_v'])
    best_accept_k = max(k_values, key=lambda k: abs(results[k]['accept_rate'] - 0.6))
    
    for k in k_values:
        r = results[k]
        print(f"k = {k}:")
        print(f"  Acceptance Rate: {r['accept_rate']:.3f}")
        print(f"  Final Step Size: {r['final_step_size']:.4f}")
        print(f"  ESS(v): {r['ess_v']:.1f}")
        print(f"  τ_int(v): {r['tau_int_v']:.2f}")
        print(f"  Sample Mean(v): {r['v_mean']:.3f} (True: 0.000)")
        print(f"  Sample Std(v): {r['v_std']:.3f} (True: 3.000)")
        print()
    
    print(f"Best ESS: k = {best_ess_k} (ESS = {results[best_ess_k]['ess_v']:.1f})")
    print(f"Best Accept Rate: k = {best_accept_k} (Rate = {results[best_accept_k]['accept_rate']:.3f})")
    
    return results

if __name__ == "__main__":
    # Run the test
    results = test_funnel_k_values()