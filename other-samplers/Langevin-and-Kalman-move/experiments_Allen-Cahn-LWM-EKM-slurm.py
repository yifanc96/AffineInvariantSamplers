import numpy as np
import matplotlib.pyplot as plt
import time
import os
import wandb


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
        tau = tau_reduced * (2**k)
    
    # Calculate effective sample size using original series length
    ess = len(orig_x) / tau
    
    return tau, acf, ess


import numpy as np

def langevin_walk_move_ensemble_dual_avg(gradient_func, potential_func, initial, n_samples, n_chains_per_group=5, 
                                        h=0.01, n_thin=1, target_accept=0.57, n_warmup=1000,
                                        gamma=0.05, t0=10, kappa=0.75):
    """
    Vectorized Langevin Walk Move sampler using normalized ensemble preconditioning with dual averaging
    for automatic step size adaptation.
    
    This version follows the mathematical description exactly and includes dual averaging to tune
    the step size to achieve a target acceptance rate during warmup.
    
    Parameters:
    -----------
    gradient_func : callable
        Function that computes the gradient of the potential V(x)
    potential_func : callable
        Function that computes the potential V(x)
    initial : np.ndarray
        Initial state
    n_samples : int
        Number of samples to collect (after warmup)
    n_chains_per_group : int
        Number of chains per group (default: 5)
    h : float
        Initial step size (default: 0.01)
    n_thin : int
        Thinning factor - store every n_thin sample (default: 1, no thinning)
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
    
    # Create initial states with small random perturbations
    states = np.tile(initial.flatten(), (total_chains, 1)) + 0.1 * np.random.randn(total_chains, flat_dim)
    
    # Split into two groups
    group1 = slice(0, n_chains_per_group)
    group2 = slice(n_chains_per_group, total_chains)
    
    # Dual averaging initialization
    log_h = np.log(h)
    log_h_bar = 0.0
    h_bar = 0.0
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
    
    # Main sampling loop
    for i in range(total_iterations):
        is_warmup = i < n_warmup
        current_h = h if not is_warmup else np.exp(log_h)
        
        # Store current state from all chains (only during sampling phase)
        if not is_warmup and (i - n_warmup) % n_thin == 0 and sample_idx < n_samples:
            samples[:, sample_idx] = states
            sample_idx += 1
        
        # Compute normalized centered ensemble from group 2 for group 1 update
        # B_S has shape (flat_dim, n_chains_per_group)
        group2_centered = states[group2] - np.mean(states[group2], axis=0)
        B_S2 = (group2_centered / np.sqrt(n_chains_per_group)).T  # (flat_dim, n_chains_per_group)
        
        # First group update
        current_q1 = states[group1].copy()
        current_q1_reshaped = current_q1.reshape(n_chains_per_group, *orig_dim)
        current_U1 = potential_func(current_q1_reshaped)
        
        # Compute gradient
        grad1 = gradient_func(current_q1_reshaped).reshape(n_chains_per_group, -1)
        grad1 = np.nan_to_num(grad1, nan=0.0)
        
        # Generate noise in ensemble space (dimension n_chains_per_group)
        z1 = np.random.randn(n_chains_per_group, n_chains_per_group)
        
        # Langevin proposal: x_i = x_i - h * B_S * B_S^T * ∇V(x_i) + √(2h) * B_S * z_i
        # where z_i has dimension n_chains_per_group
        
        # Compute B_S^T * B_S (covariance in parameter space)
        cov_param2 = np.dot(B_S2, B_S2.T)  # (flat_dim, flat_dim)
        
        # Drift term: -h * B_S * B_S^T * ∇V
        drift_term = -current_h * (cov_param2 @ grad1.T).T  # (n_chains_per_group, flat_dim)
        
        # Noise term: √(2h) * B_S * z
        noise_term = np.sqrt(2 * current_h) * (B_S2 @ z1.T).T  # (n_chains_per_group, flat_dim)
        
        proposed_q1 = current_q1 + drift_term + noise_term
        
        # Compute proposed energy
        proposed_q1_reshaped = proposed_q1.reshape(n_chains_per_group, *orig_dim)
        proposed_U1 = potential_func(proposed_q1_reshaped)
        
        # Metropolis-Hastings acceptance with correct proposal probabilities
        grad1_proposed = gradient_func(proposed_q1_reshaped).reshape(n_chains_per_group, -1)
        grad1_proposed = np.nan_to_num(grad1_proposed, nan=0.0)
        
        # Forward and reverse proposal means
        mean_forward = current_q1 - current_h * (cov_param2 @ grad1.T).T
        mean_reverse = proposed_q1 - current_h * (cov_param2 @ grad1_proposed.T).T
        
        # Compute residuals
        residual_forward = proposed_q1 - mean_forward
        residual_reverse = current_q1 - mean_reverse
        
        # Compute quadratic forms with covariance 2h * B_S * B_S^T
        # The inverse covariance is (2h * B_S * B_S^T)^(-1) = (1/2h) * (B_S * B_S^T)^(-1)
        
        try:
            # Add regularization for numerical stability
            reg_cov2 = cov_param2 + 1e-8 * np.eye(flat_dim)
            L2 = np.linalg.cholesky(reg_cov2)
            
            # Solve for quadratic forms: (1/2h) * residual^T * (B_S * B_S^T)^(-1) * residual
            Y_forward = np.linalg.solve(L2, residual_forward.T)  # (flat_dim, n_chains_per_group)
            Y_reverse = np.linalg.solve(L2, residual_reverse.T)  # (flat_dim, n_chains_per_group)
            
            # Quadratic forms: (1/(4h)) * residual^T * (B_S * B_S^T)^(-1) * residual
            # Note: factor is 1/(4h), not 1/(2h), to match covariance 2h * B_S * B_S^T
            log_q_forward = -np.sum(Y_forward**2, axis=0) / (4 * current_h)
            log_q_reverse = -np.sum(Y_reverse**2, axis=0) / (4 * current_h)
            
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            inv_cov2 = np.linalg.pinv(cov_param2)
            
            log_q_forward = np.zeros(n_chains_per_group)
            log_q_reverse = np.zeros(n_chains_per_group)
            
            for j in range(n_chains_per_group):
                log_q_forward[j] = -residual_forward[j] @ inv_cov2 @ residual_forward[j] / (4 * current_h)
                log_q_reverse[j] = -residual_reverse[j] @ inv_cov2 @ residual_reverse[j] / (4 * current_h)
        
        # Metropolis-Hastings ratio
        dU1 = proposed_U1 - current_U1
        log_ratio = -dU1 + log_q_reverse - log_q_forward
        
        # Accept/reject with numerical stability
        accept_probs1 = np.ones_like(log_ratio)
        exp_needed = log_ratio < 0
        if np.any(exp_needed):
            safe_log_ratio = np.clip(log_ratio[exp_needed], -100, None)
            accept_probs1[exp_needed] = np.exp(safe_log_ratio)
        
        accepts1 = np.random.random(n_chains_per_group) < accept_probs1
        states[group1][accepts1] = proposed_q1[accepts1]
        
        # Track acceptances
        if is_warmup:
            accepts_warmup[group1] += accepts1
        else:
            accepts_sampling[group1] += accepts1
        
        # Second group update using ensemble from group 1
        group1_centered = states[group1] - np.mean(states[group1], axis=0)
        B_S1 = (group1_centered / np.sqrt(n_chains_per_group)).T  # (flat_dim, n_chains_per_group)
        
        current_q2 = states[group2].copy()
        current_q2_reshaped = current_q2.reshape(n_chains_per_group, *orig_dim)
        current_U2 = potential_func(current_q2_reshaped)
        
        grad2 = gradient_func(current_q2_reshaped).reshape(n_chains_per_group, -1)
        grad2 = np.nan_to_num(grad2, nan=0.0)
        
        z2 = np.random.randn(n_chains_per_group, n_chains_per_group)
        
        cov_param1 = np.dot(B_S1, B_S1.T)  # (flat_dim, flat_dim)
        
        drift_term = -current_h * (cov_param1 @ grad2.T).T
        noise_term = np.sqrt(2 * current_h) * (B_S1 @ z2.T).T
        
        proposed_q2 = current_q2 + drift_term + noise_term
        
        proposed_q2_reshaped = proposed_q2.reshape(n_chains_per_group, *orig_dim)
        proposed_U2 = potential_func(proposed_q2_reshaped)
        
        grad2_proposed = gradient_func(proposed_q2_reshaped).reshape(n_chains_per_group, -1)
        grad2_proposed = np.nan_to_num(grad2_proposed, nan=0.0)
        
        mean_forward = current_q2 - current_h * (cov_param1 @ grad2.T).T
        mean_reverse = proposed_q2 - current_h * (cov_param1 @ grad2_proposed.T).T
        
        residual_forward = proposed_q2 - mean_forward
        residual_reverse = current_q2 - mean_reverse
        
        try:
            reg_cov1 = cov_param1 + 1e-8 * np.eye(flat_dim)
            L1 = np.linalg.cholesky(reg_cov1)
            
            Y_forward = np.linalg.solve(L1, residual_forward.T)
            Y_reverse = np.linalg.solve(L1, residual_reverse.T)
            
            log_q_forward = -np.sum(Y_forward**2, axis=0) / (4 * current_h)
            log_q_reverse = -np.sum(Y_reverse**2, axis=0) / (4 * current_h)
            
        except np.linalg.LinAlgError:
            inv_cov1 = np.linalg.pinv(cov_param1)
            
            log_q_forward = np.zeros(n_chains_per_group)
            log_q_reverse = np.zeros(n_chains_per_group)
            
            for j in range(n_chains_per_group):
                log_q_forward[j] = -residual_forward[j] @ inv_cov1 @ residual_forward[j] / (4 * current_h)
                log_q_reverse[j] = -residual_reverse[j] @ inv_cov1 @ residual_reverse[j] / (4 * current_h)
        
        dU2 = proposed_U2 - current_U2
        log_ratio = -dU2 + log_q_reverse - log_q_forward
        
        accept_probs2 = np.ones_like(log_ratio)
        exp_needed = log_ratio < 0
        if np.any(exp_needed):
            safe_log_ratio = np.clip(log_ratio[exp_needed], -100, None)
            accept_probs2[exp_needed] = np.exp(safe_log_ratio)
        
        accepts2 = np.random.random(n_chains_per_group) < accept_probs2
        states[group2][accepts2] = proposed_q2[accepts2]
        
        # Track acceptances
        if is_warmup:
            accepts_warmup[group2] += accepts2
        else:
            accepts_sampling[group2] += accepts2
        
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
            h = np.exp(log_h_bar)
            print(f"Warmup complete. Final adapted step size: {h:.6f}")
    
    # Reshape final samples to original dimensions
    samples = samples.reshape((total_chains, n_samples) + orig_dim)
    
    # Compute acceptance rates for sampling phase only
    acceptance_rates = accepts_sampling / total_sampling_iterations
    
    return samples, acceptance_rates, np.array(step_size_history)


    
import numpy as np

def ensemble_kalman_move_dual_avg(forward_func, initial, n_samples, M=None,
                                 n_chains_per_group=5, h=0.01, n_thin=1, use_metropolis=True,
                                 target_accept=0.57, n_warmup=1000, gamma=0.05, t0=10, kappa=0.75):
    """
    Ensemble Kalman Move sampler for least squares type densities with dual averaging
    for automatic step size adaptation.
    
    For density π(x) ∝ exp(-V(x)) with V(x) = ½G(x)ᵀMG(x) where G: ℝᵈ → ℝʳ.
    
    The proposal is: x' = x - h * B_S * F_S^T * M * G(x) + √(2h) * B_S * z
    where:
    - B_S: normalized centered ensemble from other group (flat_dim, n_chains_per_group)
    - F_S: normalized centered G(x) from other group (data_dim, n_chains_per_group)  
    - z ~ N(0, I_{n_chains_per_group × n_chains_per_group})
    
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
    
    # Create initial states with small random perturbations
    states = np.tile(initial.flatten(), (total_chains, 1)) + 0.1 * np.random.randn(total_chains, flat_dim)
    
    # Split into two groups
    group1 = slice(0, n_chains_per_group)
    group2 = slice(n_chains_per_group, total_chains)
    
    # Dual averaging initialization
    log_h = np.log(h)
    log_h_bar = 0.0
    h_bar = 0.0
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
    
    # Determine data dimension from first forward model evaluation
    test_G = forward_func(initial)
    data_dim = len(test_G)
    
    # Set default M = I if not provided
    if M is None:
        M = np.eye(data_dim)
    
    # Main sampling loop
    for i in range(total_iterations):
        is_warmup = i < n_warmup
        current_h = h if not is_warmup else np.exp(log_h)
        
        # Store current state from all chains (only during sampling phase)
        if not is_warmup and (i - n_warmup) % n_thin == 0 and sample_idx < n_samples:
            samples[:, sample_idx] = states
            sample_idx += 1
        
        # Update group 1 using group 2 for ensemble information
        group2_reshaped = states[group2].reshape(n_chains_per_group, *orig_dim)
        G_group2 = forward_func(group2_reshaped)  # (n_chains_per_group, data_dim)
        mean_G2 = np.mean(G_group2, axis=0)  # (data_dim,)
        
        # F_S1 = (1/√(N/2)) * [G(x_group2) - mean_G2]^T ∈ ℝ^{data_dim × n_chains_per_group}
        F_S1 = ((G_group2 - mean_G2) / np.sqrt(n_chains_per_group)).T  # (data_dim, n_chains_per_group)
        
        # B_S1 from group 2: normalized centered ensemble in parameter space
        group2_centered = states[group2] - np.mean(states[group2], axis=0)
        B_S1 = (group2_centered / np.sqrt(n_chains_per_group)).T  # (flat_dim, n_chains_per_group)
        
        # Vectorized update for group 1
        current_q1 = states[group1].copy()
        current_q1_reshaped = current_q1.reshape(n_chains_per_group, *orig_dim)
        G_current1 = forward_func(current_q1_reshaped)  # (n_chains_per_group, data_dim)
        current_U1 = 0.5 * np.sum(G_current1 * (G_current1 @ M), axis=1)  # (n_chains_per_group,)
        
        # Generate noise z ~ N(0, I_{n_chains_per_group × n_chains_per_group})
        z1 = np.random.randn(n_chains_per_group, n_chains_per_group)
        
        # Ensemble Kalman proposal:
        # x' = x - h * B_S1 * F_S1^T * M * G(x) + √(2h) * B_S1 * z
        
        # Drift term: -h * B_S1 * F_S1^T * M * G(x) (vectorized)
        MG_current1 = G_current1 @ M  # (n_chains_per_group, data_dim)
        drift_terms = -current_h * (B_S1 @ (F_S1.T @ MG_current1.T)).T  # (n_chains_per_group, flat_dim)
        
        # Noise term: √(2h) * B_S1 * z (vectorized)
        noise_terms = np.sqrt(2 * current_h) * (B_S1 @ z1.T).T  # (n_chains_per_group, flat_dim)
        
        # Proposed states (vectorized)
        proposed_q1 = current_q1 + drift_terms + noise_terms
        
        accepts1 = np.zeros(n_chains_per_group, dtype=bool)
        
        if use_metropolis:
            # Metropolis-Hastings correction
            proposed_q1_reshaped = proposed_q1.reshape(n_chains_per_group, *orig_dim)
            G_proposed1 = forward_func(proposed_q1_reshaped)
            proposed_U1 = 0.5 * np.sum(G_proposed1 * (G_proposed1 @ M), axis=1)
            
            # For correct Metropolis step, we need to compute proposal probabilities
            # The proposal has the form: x' ~ N(μ_forward, Σ) where:
            # μ_forward = x - h * B_S1 * F_S1^T * M * G(x)
            # Σ = 2h * B_S1 * B_S1^T (covariance of noise term)
            
            # IMPORTANT: F_S1 is FIXED (from group 2) during this proposal step
            # So both forward and reverse proposals use the SAME F_S1
            
            # Forward proposal mean (what we used to generate proposed_q1)
            mean_forward = current_q1 + drift_terms  # Note: drift_terms already has negative sign
            
            # Reverse proposal: proposed_q1 -> current_q1
            # Mean: proposed_q1 - h * B_S1 * F_S1^T * M * G(proposed_q1)
            # Note: Uses SAME F_S1 (from group 2), not recomputed!
            MG_proposed1 = G_proposed1 @ M
            reverse_drift = -current_h * (B_S1 @ (F_S1.T @ MG_proposed1.T)).T  # Same F_S1!
            mean_reverse = proposed_q1 + reverse_drift
            
            # Compute residuals
            residual_forward = proposed_q1 - mean_forward  # Should be = noise_terms
            residual_reverse = current_q1 - mean_reverse
            
            # Proposal covariance: The noise term √(2h) * B_S * z has covariance 2h * B_S * B_S^T
            # But for the log probability calculation, we factor out the h:
            # log q = -½ * residual^T * (2h * B_S * B_S^T)^(-1) * residual
            #       = -1/(4h) * residual^T * (B_S * B_S^T)^(-1) * residual
            # So we use cov = B_S * B_S^T (no h) and divide by 4h later
            cov_proposal = np.dot(B_S1, B_S1.T)  # (flat_dim, flat_dim) - NO h here!
            
            try:
                # Use Cholesky for numerical stability
                reg_cov = cov_proposal + 1e-8 * np.eye(flat_dim)
                L = np.linalg.cholesky(reg_cov)
                
                # Compute log proposal probabilities: log q(x' | x) = -½(x' - μ)^T Σ^(-1) (x' - μ)
                # Since Σ = 2h * B_S1 * B_S1^T, we have:
                # log q = -¼h * residual^T * (B_S1 * B_S1^T)^(-1) * residual
                
                # Solve L * y = residual^T for each chain
                Y_forward = np.linalg.solve(L, residual_forward.T)  # (flat_dim, n_chains_per_group)
                Y_reverse = np.linalg.solve(L, residual_reverse.T)   # (flat_dim, n_chains_per_group)
                
                # Log proposal probabilities (factor of 1/(4h) because Σ = 2h * ...)
                log_q_forward = -np.sum(Y_forward**2, axis=0) / (4 * current_h)
                log_q_reverse = -np.sum(Y_reverse**2, axis=0) / (4 * current_h)
                
            except np.linalg.LinAlgError:
                # If Cholesky fails, use simpler Metropolis (energy only)
                log_q_forward = np.zeros(n_chains_per_group)
                log_q_reverse = np.zeros(n_chains_per_group)
            
            # Metropolis-Hastings ratio
            dU1 = proposed_U1 - current_U1
            log_ratio = -dU1 + log_q_reverse - log_q_forward
            
            # Accept/reject with numerical stability
            accept_probs1 = np.ones_like(log_ratio)
            exp_needed = log_ratio < 0
            if np.any(exp_needed):
                safe_log_ratio = np.clip(log_ratio[exp_needed], -100, None)
                accept_probs1[exp_needed] = np.exp(safe_log_ratio)
            
            accepts1 = np.random.random(n_chains_per_group) < accept_probs1
            states[group1][accepts1] = proposed_q1[accepts1]
        else:
            # Pure Ensemble Kalman (no Metropolis correction)
            states[group1] = proposed_q1
            accepts1 = np.ones(n_chains_per_group, dtype=bool)  # Always accept
        
        # Track acceptances for group 1
        if is_warmup:
            accepts_warmup[group1] += accepts1
        else:
            accepts_sampling[group1] += accepts1
        
        # Update group 2 using group 1 (symmetric structure)
        group1_reshaped = states[group1].reshape(n_chains_per_group, *orig_dim)
        G_group1 = forward_func(group1_reshaped)
        mean_G1 = np.mean(G_group1, axis=0)
        F_S0 = ((G_group1 - mean_G1) / np.sqrt(n_chains_per_group)).T
        
        group1_centered = states[group1] - np.mean(states[group1], axis=0)
        B_S0 = (group1_centered / np.sqrt(n_chains_per_group)).T
        
        current_q2 = states[group2].copy()
        current_q2_reshaped = current_q2.reshape(n_chains_per_group, *orig_dim)
        G_current2 = forward_func(current_q2_reshaped)
        current_U2 = 0.5 * np.sum(G_current2 * (G_current2 @ M), axis=1)
        
        z2 = np.random.randn(n_chains_per_group, n_chains_per_group)
        
        MG_current2 = G_current2 @ M
        drift_terms = -current_h * (B_S0 @ (F_S0.T @ MG_current2.T)).T
        noise_terms = np.sqrt(2 * current_h) * (B_S0 @ z2.T).T
        proposed_q2 = current_q2 + drift_terms + noise_terms
        
        accepts2 = np.zeros(n_chains_per_group, dtype=bool)
        
        if use_metropolis:
            proposed_q2_reshaped = proposed_q2.reshape(n_chains_per_group, *orig_dim)
            G_proposed2 = forward_func(proposed_q2_reshaped)
            proposed_U2 = 0.5 * np.sum(G_proposed2 * (G_proposed2 @ M), axis=1)
            
            mean_forward = current_q2 + drift_terms
            
            # Reverse proposal: proposed_q2 -> current_q2  
            # Mean: proposed_q2 - h * B_S0 * F_S0^T * M * G(proposed_q2)
            # Note: Uses SAME F_S0 (from group 1), not recomputed!
            MG_proposed2 = G_proposed2 @ M
            reverse_drift = -current_h * (B_S0 @ (F_S0.T @ MG_proposed2.T)).T  # Same F_S0!
            mean_reverse = proposed_q2 + reverse_drift
            
            residual_forward = proposed_q2 - mean_forward
            residual_reverse = current_q2 - mean_reverse
            
            cov_proposal = np.dot(B_S0, B_S0.T)  # NO h here!
            
            try:
                reg_cov = cov_proposal + 1e-8 * np.eye(flat_dim)
                L = np.linalg.cholesky(reg_cov)
                
                Y_forward = np.linalg.solve(L, residual_forward.T)
                Y_reverse = np.linalg.solve(L, residual_reverse.T)
                
                log_q_forward = -np.sum(Y_forward**2, axis=0) / (4 * current_h)
                log_q_reverse = -np.sum(Y_reverse**2, axis=0) / (4 * current_h)
                
            except np.linalg.LinAlgError:
                log_q_forward = np.zeros(n_chains_per_group)
                log_q_reverse = np.zeros(n_chains_per_group)
            
            dU2 = proposed_U2 - current_U2
            log_ratio = -dU2 + log_q_reverse - log_q_forward
            
            accept_probs2 = np.ones_like(log_ratio)
            exp_needed = log_ratio < 0
            if np.any(exp_needed):
                safe_log_ratio = np.clip(log_ratio[exp_needed], -100, None)
                accept_probs2[exp_needed] = np.exp(safe_log_ratio)
            
            accepts2 = np.random.random(n_chains_per_group) < accept_probs2
            states[group2][accepts2] = proposed_q2[accepts2]
        else:
            states[group2] = proposed_q2
            accepts2 = np.ones(n_chains_per_group, dtype=bool)
        
        # Track acceptances for group 2
        if is_warmup:
            accepts_warmup[group2] += accepts2
        else:
            accepts_sampling[group2] += accepts2
        
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
            h = np.exp(log_h_bar)
            print(f"Warmup complete. Final adapted step size: {h:.6f}")
    
    # Reshape final samples to original dimensions
    samples = samples.reshape((total_chains, n_samples) + orig_dim)
    
    # Compute acceptance rates for sampling phase only
    acceptance_rates = accepts_sampling / total_sampling_iterations
    
    return samples, acceptance_rates, np.array(step_size_history)

def benchmark_samplers_allen_cahn(N=100, n_samples=10000, burn_in=1000, n_thin=1, save_dir=None):
    """
    Benchmark different MCMC samplers on the invariant measure of the Allen-Cahn SPDE.
    
    The Allen-Cahn SPDE has the invariant measure with density proportional to:
    exp(-∫[1/(2h) * (du/dx)² + V(u)] dx)
    
    where V(u) = (1 - u²)² is the double-well potential and h is the discretization step.
    """
    # Define discretization parameters
    h = 1.0 / N
    dim = N + 1  # Including boundary points
    
    # Define potential function V'(u) = -4u(1 - u^2)
    def V_prime(u):
        return -4 * u * (1 - u**2)
    
    # Define the gradient of the negative log density - vectorized
    def gradient(u):
        """Numerically stable vectorized gradient of the negative log density"""
        if u.ndim == 1:
            u = u.reshape(1, -1)
            
        grad = np.zeros_like(u)
        
        # Handle interior points (j=1 to j=N-1) with vectorization
        u_prev = u[:, :-2]  # u[j-1] for j=1...N-1
        u_curr = u[:, 1:-1]  # u[j] for j=1...N-1
        u_next = u[:, 2:]    # u[j+1] for j=1...N-1
        
        # Coupling term contribution: (2*u[j] - u[j-1] - u[j+1])/h
        coupling_term = (2 * u_curr - u_prev - u_next) / h
        
        # Potential term contribution
        avg_prev = (u_curr + u_prev) / 2
        avg_next = (u_curr + u_next) / 2
        
        v_prime_prev = -4 * avg_prev * (1 - avg_prev**2)
        v_prime_next = -4 * avg_next * (1 - avg_next**2)
        
        potential_term = h * (v_prime_prev + v_prime_next) / 4
        
        # Combine contributions for interior points
        grad[:, 1:-1] = coupling_term + potential_term
        
        # Handle boundary points
        u_first = u[:, 0]
        u_second = u[:, 1]
        grad[:, 0] = (u_first - u_second) / h + h * V_prime(u_first) / 4
        
        u_last = u[:, -1]
        u_second_last = u[:, -2]
        grad[:, -1] = (u_last - u_second_last) / h + h * V_prime(u_last) / 4
        
        return grad
    
    # Define the potential energy function - vectorized
    def potential(u):
        """Numerically stable vectorized negative log density (potential energy)"""
        if u.ndim == 1:
            u = u.reshape(1, -1)
            
        u_right = u[:, 1:]
        u_left = u[:, :-1]
        diffs = u_right - u_left
        
        coupling_term = np.sum(diffs**2, axis=1) / (2*h)
        
        u_avg = (u_right + u_left) / 2
        v_values = (1 - u_avg**2)**2
        potential_term = np.sum(h * v_values / 2, axis=1)
        
        total_potential = np.clip(coupling_term + potential_term, -1e10, 1e10)
            
        return total_potential

    # def forward_func(u_batch):
    #     """
    #     Forward model G(u) for Allen-Cahn SPDE where V(u) = ½|G(u)|²
        
    #     The Allen-Cahn energy is: ∫[½(∂ₓu)² + (1-u²)²] dx
    #     We can write this as ½|G(u)|² where G(u) has components:
    #     - Gradient contributions: (u[j+1] - u[j])/√h for j=0..N-1  
    #     - Potential contributions: √(2h)(1-u[j]²) for j=0..N
        
    #     This gives equal weight to gradient and potential terms in the energy.
    #     """
    #     if u_batch.ndim == 1:
    #         u_batch = u_batch.reshape(1, -1)
        
    #     batch_size = u_batch.shape[0]
        
    #     # Gradient contributions: (u[j+1] - u[j])/√h for j=0..N-1
    #     # This discretizes ∫(∂ₓu)² dx ≈ Σ (u[j+1] - u[j])²/h
    #     u_diffs = u_batch[:, 1:] - u_batch[:, :-1]  # (batch_size, dim-1)
    #     grad_part = u_diffs / np.sqrt(h)  # Scale by 1/√h
        
    #     # Potential contributions: √(2h)(1-u[j]²) for j=0..N  
    #     # This discretizes ∫2(1-u²)² dx ≈ Σ 2h(1-u[j]²)²
    #     nonlinear_part = np.sqrt(2*h) * (1 - u_batch**2)  # (batch_size, dim)
        
    #     # Concatenate: G(u) has dim-1 + dim = 2*dim-1 components
    #     G_u = np.concatenate([grad_part, nonlinear_part], axis=1)
        
    #     return G_u

    def forward_func(u_batch):
        """
        Forward model G(u) for Allen-Cahn SPDE where V(u) = ½|G(u)|²
        
        Must be consistent with potential function which uses trapezoidal rule:
        - Gradient contributions: (u[j+1] - u[j])/√h 
        - Potential contributions: √(h)(1-u_avg[j]²) where u_avg[j] = (u[j+1]+u[j])/2
        """
        if u_batch.ndim == 1:
            u_batch = u_batch.reshape(1, -1)
        
        # Gradient contributions: (u[j+1] - u[j])/√h for j=0..N-1
        u_diffs = u_batch[:, 1:] - u_batch[:, :-1]  # (batch_size, dim-1)
        grad_part = u_diffs / np.sqrt(h)
        
        # Potential contributions: √(h)(1-u_avg[j]²) for j=0..N-1
        # Use midpoint rule to match potential function
        u_left = u_batch[:, :-1]   # u[j] for j=0..N-1
        u_right = u_batch[:, 1:]   # u[j+1] for j=0..N-1
        u_avg = (u_left + u_right) / 2  # Midpoint values
        nonlinear_part = np.sqrt(h) * (1 - u_avg**2)  # (batch_size, dim-1)
        
        # Concatenate: G(u) has (dim-1) + (dim-1) = 2*(dim-1) components
        G_u = np.concatenate([grad_part, nonlinear_part], axis=1)
        
        return G_u
    
    # Function to compute path integral consistently
    def compute_path_integral(path):
        """Efficiently calculate the path integral using vectorized operations"""
        if path.ndim > 1:
            return np.array([compute_path_integral(p) for p in path])
            
        # For a single path
        left_points = path[:-1]
        right_points = path[1:]
        segment_areas = h * (left_points + right_points) / 2
        return np.sum(segment_areas)
    
    # For EKM, M should be identity
    data_dim = 2 * (dim - 1)
    M = np.eye(data_dim)
    
    # Initial state - start near one of the stable states
    initial = np.ones(dim) * 0.8 + 0.1 * np.random.randn(dim)
    
    # Dictionary to store results
    results = {}
    
    # Define samplers to benchmark
    samplers = {
        "Langevin Walk Move Ensemble": lambda: langevin_walk_move_ensemble_dual_avg(
            gradient_func=gradient, potential_func=potential, initial=initial, 
            n_samples=n_samples, n_warmup=burn_in, 
            n_chains_per_group=max(10, dim), h=1.362/(dim**(1/2)), 
            target_accept=0.57, n_thin=n_thin),
        "Ensemble Kalman Move": lambda: ensemble_kalman_move_dual_avg(
            forward_func=forward_func, M=M, initial=initial, 
            n_samples=n_samples, n_warmup=burn_in,
            n_chains_per_group=max(10, dim), h=1.362/(dim**(1/2)), 
            target_accept=0.57, n_thin=n_thin, use_metropolis=True),
    }
    
    for name, sampler_func in samplers.items():
        print(f"Running {name}...")
        start_time = time.time()
        
        try:
            samples, acceptance_rates, step_size_history = sampler_func()
            elapsed = time.time() - start_time
            
            # samples shape: (n_chains, n_samples, dim)
            post_burn_in_samples = samples
            
            # Flatten samples from all chains
            flat_samples = post_burn_in_samples.reshape(-1, dim)
            
            # Compute sample statistics
            sample_mean = np.mean(flat_samples, axis=0)
            
            # Calculate path integrals for all samples
            path_integrals = compute_path_integral(flat_samples)
            mean_path_integral = np.mean(path_integrals)
            path_integral_std = np.std(path_integrals)
            
            # Compute potential energies
            potential_energies = potential(flat_samples)
            mean_potential = np.mean(potential_energies)
            potential_var = np.var(potential_energies)
            
            # Check well mixing
            positive_well = np.mean(path_integrals > 0.5)
            negative_well = np.mean(path_integrals < -0.5)
            well_mixing = min(positive_well, negative_well)
            
            # Compute autocorrelation for path integral - following reference pattern
            # Average over chains first, then compute path integrals
            path_integrals_chain1 = compute_path_integral(np.mean(post_burn_in_samples, axis=0))
            acf = autocorrelation_fft(path_integrals_chain1)
            
            # Compute integrated autocorrelation time
            try:
                tau, _, ess = integrated_autocorr_time(path_integrals_chain1)
            except:
                tau, ess = np.nan, np.nan
                print("  Warning: Could not compute integrated autocorrelation time")
            
            # Measure fraction of time spent in positive well
            positive_fraction = np.mean(flat_samples > 0)
            
        except Exception as e:
            print(f"  Error with {name}: {str(e)}")
            # Create dummy data in case of error
            flat_samples = np.zeros((10, dim))
            acceptance_rates = np.zeros(2)
            sample_mean = np.zeros(dim)
            mean_path_integral = np.nan
            path_integral_std = np.nan
            mean_potential = np.nan
            potential_var = np.nan
            well_mixing = np.nan
            positive_fraction = np.nan
            acf = np.zeros(100)
            tau, ess = np.nan, np.nan
            elapsed = time.time() - start_time
        
        # Store results
        results[name] = {
            "samples": flat_samples,
            "acceptance_rates": acceptance_rates,
            "sample_mean": sample_mean,
            "path_integral_mean": mean_path_integral,
            "path_integral_std": path_integral_std,
            "mean_potential": mean_potential,
            "potential_var": potential_var,
            "well_mixing": well_mixing,
            "positive_fraction": positive_fraction,
            "autocorrelation": acf,
            "tau": tau,
            "ess": ess,
            "time": elapsed
        }
        
        print(f"  Acceptance rate: {np.mean(acceptance_rates):.2f}")
        print(f"  Path integral mean: {mean_path_integral:.4f}")
        print(f"  Path integral std: {path_integral_std:.4f}")
        print(f"  Well mixing rate: {well_mixing:.4f}")
        print(f"  Integrated autocorrelation time: {tau:.2f}" if np.isfinite(tau) else "  Integrated autocorrelation time: NaN")
        print(f"  Time: {elapsed:.2f} seconds")

        if save_dir:
            np.save(os.path.join(save_dir, f"samples_{name}_allen_cahn.npy"), samples)
            np.save(os.path.join(save_dir, f"acf_{name}_allen_cahn.npy"), acf)
            
    return results

n_samples = 5000
burn_in = 10**3
array_dim = [512,1024]
n_thin = 10


home = "/scratch/yc3400/AffineInvariant/"
timestamp = time.strftime("%Y%m%d-%H%M%S")
folder = f"benchmark_results_LWM_EKM_Allen-Cahn_5000samples-thin10_{timestamp}"

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
    results = benchmark_samplers_allen_cahn(
    N=dim, 
    n_samples=n_samples, 
    burn_in=burn_in, 
    n_thin=n_thin,
    save_dir=save_dir
)
