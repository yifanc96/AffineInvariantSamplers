
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def ensemble_kalman_move_k_subset(forward_func, initial, n_samples, k=None, M=None,
                                             n_chains_per_group=5, h=0.01, n_thin=1, use_metropolis=True,
                                             target_accept=0.57, n_warmup=1000, gamma=0.05, t0=10, kappa=0.75):
    """
    Ensemble Kalman Move sampler using k-subset of complementary ensemble for proposals.
    
    This vectorized implementation ensures that each chain gets a different random
    subset of k chains from the complementary ensemble for its proposal.
    
    For density π(x) ∝ exp(-V(x)) with V(x) = ½G(x)ᵀMG(x) where G: ℝᵈ → ℝʳ.
    
    The proposal uses k randomly selected chains from the complementary group:
    x' = x - h * B_S * F_S^T * M * G(x) + √(2h) * B_S * z
    where:
    - B_S: normalized centered ensemble from k selected chains (flat_dim, k)
    - F_S: normalized centered G(x) from k selected chains (data_dim, k)  
    - z ~ N(0, I_{k × k})
    
    Parameters:
    -----------
    forward_func : callable
        Function G(x) that maps parameters to data space ℝᵈ → ℝʳ
        Must accept batch input: G(x_batch) where x_batch has shape (n_batch, *param_shape)
        Returns array of shape (n_batch, data_dim)
    initial : np.ndarray
        Initial state
    n_samples : int
        Number of samples to collect (after warmup)
    k : int, optional
        Number of chains to use from complementary ensemble (default: n_chains_per_group)
        Must satisfy 2 <= k <= n_chains_per_group
    M : np.ndarray, optional
        Precision matrix in data space (default: identity)
    n_chains_per_group : int
        Number of chains per group (default: 5)
    h : float
        Initial step size (default: 0.01)
    n_thin : int
        Thinning factor (default: 1, no thinning)
    use_metropolis : bool
        Whether to use Metropolis correction for exact sampling (default: True)
    target_accept : float
        Target acceptance rate for dual averaging (default: 0.57)
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
    samples : np.ndarray
        Collected samples from all chains (after warmup)
    acceptance_rates : np.ndarray
        Final acceptance rates for all chains
    step_size_history : np.ndarray
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
    log_h = np.log(h)
    log_h_bar = 0.0
    h_bar = 0.0  # FIX: Initialize h_bar for dual averaging
    step_size_history = []
    
    # Calculate total iterations needed based on thinning factor
    total_sampling_iterations = n_samples * n_thin
    total_iterations = n_warmup + total_sampling_iterations
    
    # Storage for samples and acceptance tracking
    samples = np.zeros((total_chains, n_samples, flat_dim))
    accepts_warmup = np.zeros(total_chains)
    accepts_sampling = np.zeros(total_chains)
    sample_idx = 0
    
    # Determine data dimension from first forward model evaluation
    test_G = forward_func(initial)
    data_dim = len(test_G)
    
    # Set default M = I if not provided
    if M is None:
        M = np.eye(data_dim)
    
    print(f"Using a unique random k-subset for each chain (k={k})")
    
    # Initial forward model evaluations for both groups
    group1_reshaped = states[group1_idx].reshape(n_chains_per_group, *orig_dim)
    group2_reshaped = states[group2_idx].reshape(n_chains_per_group, *orig_dim)
    G_group1 = forward_func(group1_reshaped)  # (n_chains_per_group, data_dim)
    G_group2 = forward_func(group2_reshaped)  # (n_chains_per_group, data_dim)
    
    # Main sampling loop
    for i in range(total_iterations):
        is_warmup = i < n_warmup
        current_h = h if not is_warmup else np.exp(log_h)
        
        # Store current state from all chains (only during sampling phase)
        if not is_warmup and (i - n_warmup) % n_thin == 0 and sample_idx < n_samples:
            samples[:, sample_idx] = states
            sample_idx += 1
        
        # --- Vectorized Update for Group 1 ---
        # Each chain in group 1 uses a unique random k-subset from group 2
        
        # 1. Generate random k-subsets for each chain in group 1 from group 2 chains
        selected_indices_2 = np.array([np.random.choice(n_chains_per_group, size=k, replace=False) 
                                       for _ in range(n_chains_per_group)]) # (n_cpg, k)

        # 2. Use advanced indexing to get all needed states and G values at once
        # These are now 3D arrays: (n_chains_per_group, k, dim)
        selected_states_2 = states[group2_idx][selected_indices_2] # (n_cpg, k, flat_dim)
        G_selected_2 = G_group2[selected_indices_2] # (n_cpg, k, data_dim)
        
        # 3. Centering and normalization for each k-subset using broadcasting
        mean_G_selected_2 = np.mean(G_selected_2, axis=1, keepdims=True) # (n_cpg, 1, data_dim)
        F_S1 = (G_selected_2 - mean_G_selected_2) / np.sqrt(k) # (n_cpg, k, data_dim)
        selected_centered_2 = selected_states_2 - np.mean(selected_states_2, axis=1, keepdims=True) # (n_cpg, k, flat_dim)
        B_S1 = selected_centered_2 / np.sqrt(k) # (n_cpg, k, flat_dim)

        current_q1 = states[group1_idx] # (n_cpg, flat_dim)
        G_current1 = G_group1 # (n_cpg, data_dim)
        current_U1 = 0.5 * np.sum(G_current1 * (G_current1 @ M), axis=1) # (n_cpg,)

        # 4. Generate noise z ~ N(0, I_{k}) for each chain
        z1 = np.random.randn(n_chains_per_group, k) # (n_cpg, k)
        
        # 5. Vectorized proposal calculation using einsum
        MG_current1 = G_current1 @ M # (n_cpg, data_dim)
        # Corrected einsum for B_S @ F_S.T @ M @ G_current
        drift_terms = -current_h * np.einsum('nsp,nsd,nd->np', B_S1, F_S1, MG_current1) # (n_cpg, flat_dim)
        
        # Corrected einsum for B_S @ z
        noise_terms = np.sqrt(2 * current_h) * np.einsum('nsp,ns->np', B_S1, z1) # (n_cpg, flat_dim)

        proposed_q1 = current_q1 + drift_terms + noise_terms
        
        accepts1 = np.zeros(n_chains_per_group, dtype=bool)

        if use_metropolis:
            proposed_q1_reshaped = proposed_q1.reshape(n_chains_per_group, *orig_dim)
            G_proposed1 = forward_func(proposed_q1_reshaped)
            proposed_U1 = 0.5 * np.sum(G_proposed1 * (G_proposed1 @ M), axis=1)
            
            mean_forward = current_q1 + drift_terms
            
            MG_proposed1 = G_proposed1 @ M
            reverse_drift = -current_h * np.einsum('nsp,nsd,nd->np', B_S1, F_S1, MG_proposed1)
            mean_reverse = proposed_q1 + reverse_drift
            
            residual_forward = proposed_q1 - mean_forward
            residual_reverse = current_q1 - mean_reverse
            
            # Corrected einsum for the Gram matrix: B_S^T @ B_S
            reg_gram = np.einsum('nsp,nqp->nsq', B_S1, B_S1) + 1e-12 * np.eye(k) # (n_cpg, k, k)
            
            # Log proposal probabilities
            try:
                # Corrected einsum for alpha terms
                alpha_forward = np.linalg.solve(reg_gram, np.einsum('nsp,np->ns', B_S1, residual_forward)) # (n_cpg, k)
                alpha_reverse = np.linalg.solve(reg_gram, np.einsum('nsp,np->ns', B_S1, residual_reverse)) # (n_cpg, k)
                
                log_q_forward = -np.sum(alpha_forward**2, axis=1) / (4 * current_h)
                log_q_reverse = -np.sum(alpha_reverse**2, axis=1) / (4 * current_h)
            except np.linalg.LinAlgError:
                print(f"Warning: Numerical issues with regularized Gram matrix at iteration {i} (Group 1)")
                log_q_forward = np.zeros(n_chains_per_group)
                log_q_reverse = np.zeros(n_chains_per_group)
            
            log_ratio = - (proposed_U1 - current_U1) + log_q_reverse - log_q_forward
            
            accept_probs1 = np.clip(np.exp(log_ratio), 0.0, 1.0)
            accepts1 = np.random.random(n_chains_per_group) < accept_probs1
            
            states[group1_idx][accepts1] = proposed_q1[accepts1]
            G_group1[accepts1] = G_proposed1[accepts1]
        
        else: # Pure Ensemble Kalman
            states[group1_idx] = proposed_q1
            accepts1 = np.ones(n_chains_per_group, dtype=bool)
            G_group1 = forward_func(proposed_q1.reshape(n_chains_per_group, *orig_dim))
        
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
        G_selected_1 = G_group1[selected_indices_1]
        
        mean_G_selected_1 = np.mean(G_selected_1, axis=1, keepdims=True)
        F_S2 = (G_selected_1 - mean_G_selected_1) / np.sqrt(k)
        selected_centered_1 = selected_states_1 - np.mean(selected_states_1, axis=1, keepdims=True)
        B_S2 = selected_centered_1 / np.sqrt(k)

        current_q2 = states[group2_idx]
        G_current2 = G_group2
        current_U2 = 0.5 * np.sum(G_current2 * (G_current2 @ M), axis=1)

        z2 = np.random.randn(n_chains_per_group, k)
        
        MG_current2 = G_current2 @ M
        # Corrected einsum for B_S @ F_S.T @ M @ G_current
        drift_terms = -current_h * np.einsum('nsp,nsd,nd->np', B_S2, F_S2, MG_current2)
        # Corrected einsum for B_S @ z
        noise_terms = np.sqrt(2 * current_h) * np.einsum('nsp,ns->np', B_S2, z2)

        proposed_q2 = current_q2 + drift_terms + noise_terms
        
        accepts2 = np.zeros(n_chains_per_group, dtype=bool)

        if use_metropolis:
            proposed_q2_reshaped = proposed_q2.reshape(n_chains_per_group, *orig_dim)
            G_proposed2 = forward_func(proposed_q2_reshaped)
            proposed_U2 = 0.5 * np.sum(G_proposed2 * (G_proposed2 @ M), axis=1)
            
            mean_forward = current_q2 + drift_terms
            
            MG_proposed2 = G_proposed2 @ M
            reverse_drift = -current_h * np.einsum('nsp,nsd,nd->np', B_S2, F_S2, MG_proposed2)
            mean_reverse = proposed_q2 + reverse_drift
            
            residual_forward = proposed_q2 - mean_forward
            residual_reverse = current_q2 - mean_reverse
            
            # Corrected einsum for the Gram matrix: B_S^T @ B_S
            reg_gram = np.einsum('nsp,nqp->nsq', B_S2, B_S2) + 1e-12 * np.eye(k)
            
            try:
                # Corrected einsum for alpha terms
                alpha_forward = np.linalg.solve(reg_gram, np.einsum('nsp,np->ns', B_S2, residual_forward))
                alpha_reverse = np.linalg.solve(reg_gram, np.einsum('nsp,np->ns', B_S2, residual_reverse))
                
                log_q_forward = -np.sum(alpha_forward**2, axis=1) / (4 * current_h)
                log_q_reverse = -np.sum(alpha_reverse**2, axis=1) / (4 * current_h)
            except np.linalg.LinAlgError:
                print(f"Warning: Numerical issues with regularized Gram matrix at iteration {i} (Group 2)")
                log_q_forward = np.zeros(n_chains_per_group)
                log_q_reverse = np.zeros(n_chains_per_group)
            
            log_ratio = - (proposed_U2 - current_U2) + log_q_reverse - log_q_forward
            
            accept_probs2 = np.clip(np.exp(log_ratio), 0.0, 1.0)
            accepts2 = np.random.random(n_chains_per_group) < accept_probs2
            
            states[group2_idx][accepts2] = proposed_q2[accepts2]
            G_group2[accepts2] = G_proposed2[accepts2]
        
        else: # Pure Ensemble Kalman
            states[group2_idx] = proposed_q2
            accepts2 = np.ones(n_chains_per_group, dtype=bool)
            G_group2 = forward_func(proposed_q2.reshape(n_chains_per_group, *orig_dim))
        
        # Track acceptances
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
            h = np.exp(log_h)
            print(f"Warmup complete. Final adapted step size: {h:.6f}")
    
    # Reshape final samples to original dimensions
    samples = samples.reshape((total_chains, n_samples) + orig_dim)
    
    # Compute acceptance rates for sampling phase only
    acceptance_rates = accepts_sampling / total_sampling_iterations
    
    return samples, acceptance_rates, np.array(step_size_history)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

# Assume the EKM k-subset function is imported
# from ekm_k_subset import ensemble_kalman_move_k_subset

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
        tau_reduced = tau_reduced * (2**k)
    
    # Calculate effective sample size using original series length
    ess = len(orig_x) / tau
    
    return tau, acf, ess

def funnel_log_prob(x):
    """
    Neal's funnel distribution log probability.
    First dimension determines the variance of other dimensions.
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
         
    v = x[:, 0]  # First dimension is v
    log_prob_v = -0.5 * v**2/9  # Prior on v ~ N(0, 9)
         
    # Other dimensions have variance exp(v)
    x_given_v = x[:, 1:]
    d = x_given_v.shape[1]
    log_prob_x_given_v = -0.5 * np.sum(x_given_v**2 / np.exp(v).reshape(-1, 1), axis=1) - 0.5 * d * v
          
    return log_prob_v + log_prob_x_given_v

def funnel_forward_map(x):
    """
    Convert funnel distribution to exact least squares form.
    
    From funnel_log_prob:
    log π(x) = -0.5 * v²/9 - 0.5 * sum(x_i² / exp(v)) - 0.5 * d * v
    
    Completing the square for v terms:
    -0.5 * v²/9 - 0.5 * d * v = -0.5 * (v/3 + 3d/2)² + constant
    
    So: log π(x) = -0.5 * sum(x_i² * exp(-v)) - 0.5 * (v/3 + 3d/2)² + constant
                 = -0.5 * ||G(x)||² + constant
    
    where G(x) = [x_1 * exp(-v/2), ..., x_d * exp(-v/2), v/3 + 3d/2]
    """
    single_input = False
    if x.ndim == 1:
        x = x.reshape(1, -1)
        single_input = True
    
    n_batch, n_dim = x.shape
    v = x[:, 0]  # First dimension
    x_rest = x[:, 1:]  # Remaining dimensions
    d = x_rest.shape[1]
    
    # Construct residual vector G(x)
    G_list = []
    
    # x_i * exp(-v/2) terms (from -0.5 * sum(x_i² * exp(-v)) = -0.5 * sum((x_i * exp(-v/2))²))
    exp_neg_v_half = np.exp(-v / 2).reshape(-1, 1)  # Shape: (n_batch, 1)
    scaled_x = x_rest * exp_neg_v_half  # Shape: (n_batch, d)
    G_list.append(scaled_x)
    
    # v/3 + 3d/2 term (from completing the square: -0.5 * (v/3 + 3d/2)²)
    v_completed_square = (v / 3 + 1.5 * d).reshape(-1, 1)  # Shape: (n_batch, 1)
    G_list.append(v_completed_square)
    
    # Concatenate all terms
    G = np.concatenate(G_list, axis=1)  # Shape: (n_batch, d + 1)
    
    # If single input, return 1D array for consistency with EKM's dimension detection
    if single_input:
        G = G.squeeze(0)  # Remove batch dimension: (d + 1,)
    
    return G

def compute_autocorrelation(chain, max_lag=50):
    """Compute autocorrelation function up to max_lag."""
    return autocorrelation_fft(chain, max_lag)

def effective_sample_size(chain, max_lag=None):
    """Estimate effective sample size using autocorrelation."""
    tau_int, acf, ess = integrated_autocorr_time(chain, M=5, c=10)
    return ess, tau_int

def test_funnel_k_values():
    """Test EKM with different k values on the funnel distribution."""
    
    # Problem setup
    dim = 5  # Total dimensions (v + 9 other dimensions)
    initial = np.zeros(dim)
    initial[0] = 0.0  # Start v near 0
    initial[1:] = 0.1 * np.random.randn(dim-1)
    
    n_samples = 400000
    n_warmup = 40000
    n_chains_per_group = 4*dim
    
    # Different k values to test
    # k_values = [2,n_chains_per_group]  # k=n_chains_per_group is full ensemble (original EKM)
    k_values = [2, 5, 10]
    results = {}
    
    print("Testing funnel distribution with different k values...")
    print(f"Dimension: {dim}, Chains per group: {n_chains_per_group}")
    print("-" * 60)
    
    for k in k_values:
        print(f"Running EKM with k={k}...")
        
        # Run EKM with current k value
        samples, accept_rates, step_history = ensemble_kalman_move_k_subset(
            forward_func=funnel_forward_map,
            initial=initial,
            n_samples=n_samples,
            k=k,
            M=None,  # Use identity
            n_chains_per_group=n_chains_per_group,
            h=0.01,  # Initial step size
            n_thin=1,
            use_metropolis=True,
            target_accept=0.2,
            n_warmup=n_warmup,
            gamma=0.05,
            t0=10,
            kappa=0.75
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
    fig.suptitle('EKM k-subset Analysis: Funnel Distribution', fontsize=16)
    
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