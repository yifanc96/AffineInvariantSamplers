
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def ensemble_kalman_move_k_subset(forward_func, initial, n_samples, k=None, M=None,
                                 n_chains_per_group=5, h=0.01, n_thin=1, use_metropolis=True,
                                 target_accept=0.57, n_warmup=1000, gamma=0.05, t0=10, kappa=0.75):
    """
    Ensemble Kalman Move sampler using k-subset of complementary ensemble for proposals.
    
    For density π(x) ∝ exp(-V(x)) with V(x) = ½G(x)ᵀMG(x) where G: ℝᵈ → ℝʳ.
    
    The proposal uses only k randomly selected chains from the complementary group:
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
    
    print(f"Using k={k} chains from complementary ensemble (out of {n_chains_per_group})")
    
    # Initial forward model evaluations for both groups
    group1_reshaped = states[group1].reshape(n_chains_per_group, *orig_dim)
    group2_reshaped = states[group2].reshape(n_chains_per_group, *orig_dim)
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
        
        # Update group 1 using k selected chains from group 2
        # Use the saved G_group2 from previous iteration (or initial computation)
        
        # Randomly select k chains from group 2
        selected_indices_2 = np.random.choice(n_chains_per_group, size=k, replace=False)
        selected_states_2 = states[group2][selected_indices_2]  # (k, flat_dim)
        
        # Use already computed G_group2, just select the k chains
        G_selected_2 = G_group2[selected_indices_2]  # (k, data_dim) - no forward call!
        mean_G_selected_2 = np.mean(G_selected_2, axis=0)  # (data_dim,)
        
        # F_S1 = (1/√k) * [G(selected_x_group2) - mean_G_selected_2]^T ∈ ℝ^{data_dim × k}
        F_S1 = ((G_selected_2 - mean_G_selected_2) / np.sqrt(k)).T  # (data_dim, k)
        
        # B_S1 from selected chains in group 2: normalized centered ensemble in parameter space
        selected_centered_2 = selected_states_2 - np.mean(selected_states_2, axis=0)
        B_S1 = (selected_centered_2 / np.sqrt(k)).T  # (flat_dim, k)
        
        # Vectorized update for group 1
        current_q1 = states[group1].copy()
        # Use saved G_group1 from previous iteration (or initial computation)
        G_current1 = G_group1.copy()  # (n_chains_per_group, data_dim) - no forward call!
        current_U1 = 0.5 * np.sum(G_current1 * (G_current1 @ M), axis=1)  # (n_chains_per_group,)
        
        # Generate noise z ~ N(0, I_{k × k})
        z1 = np.random.randn(n_chains_per_group, k)  # Each chain gets independent k-dim noise
        
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
            # Metropolis-Hastings correction with subspace noise
            proposed_q1_reshaped = proposed_q1.reshape(n_chains_per_group, *orig_dim)
            G_proposed1 = forward_func(proposed_q1_reshaped)  # Only forward call needed for group 1
            proposed_U1 = 0.5 * np.sum(G_proposed1 * (G_proposed1 @ M), axis=1)
            
            # CRITICAL: For the reverse proposal, we need the SAME selected indices
            # and the SAME F_S1, B_S1 matrices to ensure detailed balance
            
            # Forward proposal mean (what we used to generate proposed_q1)
            mean_forward = current_q1 + drift_terms  # drift_terms already has negative sign
            
            # Reverse proposal: proposed_q1 -> current_q1
            # Mean: proposed_q1 - h * B_S1 * F_S1^T * M * G(proposed_q1)
            # IMPORTANT: Uses SAME B_S1, F_S1 (from selected chains), not recomputed!
            MG_proposed1 = G_proposed1 @ M
            reverse_drift = -current_h * (B_S1 @ (F_S1.T @ MG_proposed1.T)).T  # Same B_S1, F_S1!
            mean_reverse = proposed_q1 + reverse_drift
            
            # Compute residuals
            residual_forward = proposed_q1 - mean_forward  # Should equal noise_terms
            residual_reverse = current_q1 - mean_reverse
            
            # CRITICAL: Proposal covariance is B_S1 @ B_S1^T (rank-k matrix in flat_dim space)
            # But since both proposal and reverse move only in span(B_S1), we can work in k-dim space
            
            # Key insight: If residual = B_S1 @ α for some α ∈ ℝᵏ, then
            # residual^T @ (B_S1 @ B_S1^T)^{-1} @ residual = α^T @ α
            # So we project residuals onto span(B_S1) and compute norm in k-space
            
            try:
                # Project residuals onto the k-dimensional subspace spanned by B_S1
                # Add small regularization to B_S1^T @ B_S1 for numerical stability
                reg_gram = B_S1.T @ B_S1 + 1e-12 * np.eye(k)  # k×k matrix with regularization
                alpha_forward = np.linalg.solve(reg_gram, B_S1.T @ residual_forward.T)  # (k, n_chains_per_group)
                alpha_reverse = np.linalg.solve(reg_gram, B_S1.T @ residual_reverse.T)  # (k, n_chains_per_group)
                
                # Log proposal probabilities in k-dimensional subspace
                # The small regularization slightly changes the covariance from 2h*I_k to 2h*(I_k + ε*I_k)
                # But for ε = 1e-12, this is negligible: log q ≈ -α^T @ α / (4h)
                log_q_forward = -np.sum(alpha_forward**2, axis=0) / (4 * current_h)
                log_q_reverse = -np.sum(alpha_reverse**2, axis=0) / (4 * current_h)
                
            except np.linalg.LinAlgError:
                # Even with regularization, we might have numerical issues in extreme cases
                print(f"Warning: Numerical issues with regularized Gram matrix at iteration {i}")
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
            # Update G_group1 for accepted proposals (important for next iteration!)
            if np.any(accepts1):
                if use_metropolis:  # G_proposed1 was computed above
                    G_group1[accepts1] = G_proposed1[accepts1]
                else:  # Need to compute G for new states
                    new_states_reshaped = proposed_q1[accepts1].reshape(np.sum(accepts1), *orig_dim)
                    G_group1[accepts1] = forward_func(new_states_reshaped)
        else:
            # Pure Ensemble Kalman (no Metropolis correction) - all proposals accepted
            states[group1] = proposed_q1
            accepts1 = np.ones(n_chains_per_group, dtype=bool)  # Always accept
            # Update G_group1 for all chains since all were accepted
            G_group1 = forward_func(proposed_q1.reshape(n_chains_per_group, *orig_dim))
        
        # Track acceptances for group 1
        if is_warmup:
            accepts_warmup[group1] += accepts1
        else:
            accepts_sampling[group1] += accepts1
        
        # Update group 2 using k selected chains from group 1 (symmetric structure)
        # Use the UPDATED G_group1 (some chains may have changed above)
        
        # Randomly select k chains from group 1
        selected_indices_1 = np.random.choice(n_chains_per_group, size=k, replace=False)
        selected_states_1 = states[group1][selected_indices_1]  # (k, flat_dim)
        
        # Use updated G_group1, just select the k chains
        G_selected_1 = G_group1[selected_indices_1]  # (k, data_dim) - no forward call!
        mean_G_selected_1 = np.mean(G_selected_1, axis=0)
        F_S0 = ((G_selected_1 - mean_G_selected_1) / np.sqrt(k)).T
        
        selected_centered_1 = selected_states_1 - np.mean(selected_states_1, axis=0)
        B_S0 = (selected_centered_1 / np.sqrt(k)).T
        
        current_q2 = states[group2].copy()
        # Use saved G_group2 from previous iteration (or initial computation)
        G_current2 = G_group2.copy()  # (n_chains_per_group, data_dim) - no forward call!
        current_U2 = 0.5 * np.sum(G_current2 * (G_current2 @ M), axis=1)
        
        z2 = np.random.randn(n_chains_per_group, k)
        
        MG_current2 = G_current2 @ M
        drift_terms = -current_h * (B_S0 @ (F_S0.T @ MG_current2.T)).T
        noise_terms = np.sqrt(2 * current_h) * (B_S0 @ z2.T).T
        proposed_q2 = current_q2 + drift_terms + noise_terms
        
        accepts2 = np.zeros(n_chains_per_group, dtype=bool)
        
        if use_metropolis:
            proposed_q2_reshaped = proposed_q2.reshape(n_chains_per_group, *orig_dim)
            G_proposed2 = forward_func(proposed_q2_reshaped)  # Only forward call needed for group 2
            proposed_U2 = 0.5 * np.sum(G_proposed2 * (G_proposed2 @ M), axis=1)
            
            mean_forward = current_q2 + drift_terms
            
            # Reverse proposal: proposed_q2 -> current_q2  
            # Mean: proposed_q2 - h * B_S0 * F_S0^T * M * G(proposed_q2)
            # IMPORTANT: Uses SAME B_S0, F_S0 (from selected chains), not recomputed!
            MG_proposed2 = G_proposed2 @ M
            reverse_drift = -current_h * (B_S0 @ (F_S0.T @ MG_proposed2.T)).T  # Same B_S0, F_S0!
            mean_reverse = proposed_q2 + reverse_drift
            
            residual_forward = proposed_q2 - mean_forward
            residual_reverse = current_q2 - mean_reverse
            
            cov_proposal = np.dot(B_S0, B_S0.T)  # (flat_dim, flat_dim), rank k
            
            try:
                # Project residuals onto the k-dimensional subspace spanned by B_S0
                # Add small regularization to B_S0^T @ B_S0 for numerical stability
                reg_gram = B_S0.T @ B_S0 + 1e-12 * np.eye(k)  # k×k matrix with regularization
                alpha_forward = np.linalg.solve(reg_gram, B_S0.T @ residual_forward.T)  # (k, n_chains_per_group)
                alpha_reverse = np.linalg.solve(reg_gram, B_S0.T @ residual_reverse.T)  # (k, n_chains_per_group)
                
                # Log proposal probabilities in k-dimensional subspace
                log_q_forward = -np.sum(alpha_forward**2, axis=0) / (4 * current_h)
                log_q_reverse = -np.sum(alpha_reverse**2, axis=0) / (4 * current_h)
                
            except np.linalg.LinAlgError:
                # Even with regularization, we might have numerical issues in extreme cases
                print(f"Warning: Numerical issues with regularized Gram matrix at iteration {i}")
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
            # Update G_group2 for accepted proposals (important for next iteration!)
            if np.any(accepts2):
                if use_metropolis:  # G_proposed2 was computed above
                    G_group2[accepts2] = G_proposed2[accepts2]
                else:  # Need to compute G for new states
                    new_states_reshaped = proposed_q2[accepts2].reshape(np.sum(accepts2), *orig_dim)
                    G_group2[accepts2] = forward_func(new_states_reshaped)
        else:
            # Pure Ensemble Kalman (no Metropolis correction) - all proposals accepted
            states[group2] = proposed_q2
            accepts2 = np.ones(n_chains_per_group, dtype=bool)  # Always accept
            # Update G_group2 for all chains since all were accepted
            G_group2 = forward_func(proposed_q2.reshape(n_chains_per_group, *orig_dim))
        
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
        
    def forward_func(x_batch):
        """
        Forward model G(x) = x - μ
        x_batch: (n_batch, dim)
        returns: (n_batch, dim)  # data dimension = parameter dimension
        """
        if x_batch.ndim == 1:
            x_batch = x_batch.reshape(1, -1)
        
        # G(x) = x - μ
        return x_batch - true_mean  # (n_batch, dim)
    
    # For this construction, M = Λ (precision matrix)
    M = np.diag(precision_diag)
    
    # Initial state
    initial = np.zeros(dim)
    
    # Dictionary to store results
    results = {}
    
    # Define samplers to benchmark with burn-in
    total_samples = n_samples
    
    
    # Define samplers to benchmark - adjust parameters for high-dimensional case
    samplers = {
        "Ensemble Kalman Move": lambda: ensemble_kalman_move_k_subset(forward_func=forward_func, M = M, k=2, initial=initial, n_samples=total_samples, n_warmup=burn_in, n_chains_per_group=dim, h=1.362/(dim**(1/2)), target_accept=0.57, n_thin=n_thin, use_metropolis=True),
    }
    
    for name, sampler_func in samplers.items():
        print(f"Running {name}...")
        start_time = time.time()
        samples, acceptance_rates, step_size_history = sampler_func()
        elapsed = time.time() - start_time
        
        post_burn_in_samples = samples
        
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
    

n_samples = 20000
burn_in = 10**3
array_dim = [4, 8, 16, 32]
n_thin = 1
# array_dim = [128]

print(f'n_sample{n_samples}, burn_in{burn_in}, n_thin{n_thin}')
    
for dim in array_dim:
    print(f"dim={dim}")
    save_dir = None
    
    
    # Run benchmarks and save results
    results, true_mean, cov_diag = benchmark_samplers(
    dim=dim, 
    n_samples=n_samples, 
    burn_in=burn_in, 
    condition_number=1000,
    n_thin=n_thin,
    save_dir=save_dir
)
