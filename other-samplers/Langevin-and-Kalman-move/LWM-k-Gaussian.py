import numpy as np
import time
import os


def langevin_walk_move_k_subset(gradient_func, potential_func, initial, n_samples, n_chains_per_group=5, 
                                        h=0.01, n_thin=1, target_accept=0.57, n_warmup=1000,
                                        gamma=0.05, t0=10, kappa=0.75, k=None):
    """
    Vectorized Langevin Walk Move sampler using normalized ensemble preconditioning with dual averaging
    for automatic step size adaptation, with k-subset sampling from complementary ensembles.
    
    This version follows the mathematical description exactly and includes dual averaging to tune
    the step size to achieve a target acceptance rate during warmup. Each chain uses only k randomly
    sampled chains from its complementary ensemble for preconditioning.
    
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
    k : int, optional
        Number of chains to randomly sample from complementary ensemble for each chain.
        If None, uses all chains from complementary ensemble (default: None)
    
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
    
    # Validate k parameter
    if k is None:
        k = n_chains_per_group
    elif k > n_chains_per_group:
        print(f"Warning: k={k} is larger than n_chains_per_group={n_chains_per_group}. Using k={n_chains_per_group}")
        k = n_chains_per_group
    elif k <= 0:
        raise ValueError("k must be positive")
    
    print(f"Using k={k} chains from complementary ensemble for preconditioning")
    
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
    
    def vectorized_group_update(current_states, gradients, complementary_states):
        """
        Vectorized update for a group of chains using k-subset sampling from complementary ensemble.
        
        Parameters:
        -----------
        current_states : np.ndarray (n_chains_per_group, flat_dim)
        gradients : np.ndarray (n_chains_per_group, flat_dim)  
        complementary_states : np.ndarray (n_chains_per_group, flat_dim)
        
        Returns:
        --------
        proposed_states : np.ndarray (n_chains_per_group, flat_dim)
        log_q_forward : np.ndarray (n_chains_per_group,)
        log_q_reverse : np.ndarray (n_chains_per_group,)
        """
        n_chains = current_states.shape[0]
        
        if k == n_chains_per_group:
            # Use all chains - original vectorized approach
            complementary_centered = complementary_states - np.mean(complementary_states, axis=0)
            B_S = (complementary_centered / np.sqrt(k)).T  # (flat_dim, k)
            
            # Generate noise in ensemble space
            z = np.random.randn(n_chains, k)
            
            # Compute covariance matrix
            cov_param = np.dot(B_S, B_S.T)  # (flat_dim, flat_dim)
            
            # Langevin proposal: vectorized across all chains
            drift_term = -current_h * (cov_param @ gradients.T).T  # (n_chains, flat_dim)
            noise_term = np.sqrt(2 * current_h) * (B_S @ z.T).T  # (n_chains, flat_dim)
            
            proposed_states = current_states + drift_term + noise_term
            
            # Compute proposed gradients
            proposed_reshaped = proposed_states.reshape(n_chains, *orig_dim)
            proposed_gradients = gradient_func(proposed_reshaped).reshape(n_chains, -1)
            proposed_gradients = np.nan_to_num(proposed_gradients, nan=0.0)
            
            # Compute proposal probabilities
            mean_forward = current_states - current_h * (cov_param @ gradients.T).T
            mean_reverse = proposed_states - current_h * (cov_param @ proposed_gradients.T).T
            
            residual_forward = proposed_states - mean_forward
            residual_reverse = current_states - mean_reverse
            
            try:
                # Add regularization for numerical stability
                reg_cov = cov_param + 1e-8 * np.eye(flat_dim)
                L = np.linalg.cholesky(reg_cov)
                
                # Vectorized quadratic form computation
                Y_forward = np.linalg.solve(L, residual_forward.T)  # (flat_dim, n_chains)
                Y_reverse = np.linalg.solve(L, residual_reverse.T)  # (flat_dim, n_chains)
                
                log_q_forward = -np.sum(Y_forward**2, axis=0) / (4 * current_h)
                log_q_reverse = -np.sum(Y_reverse**2, axis=0) / (4 * current_h)
                
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse
                inv_cov = np.linalg.pinv(cov_param)
                log_q_forward = -np.sum(residual_forward * (residual_forward @ inv_cov), axis=1) / (4 * current_h)
                log_q_reverse = -np.sum(residual_reverse * (residual_reverse @ inv_cov), axis=1) / (4 * current_h)
        
        else:
            # k < n_chains_per_group: use different random subsets for each chain
            # Generate random indices for each chain
            indices = np.array([np.random.choice(n_chains_per_group, size=k, replace=False) 
                              for _ in range(n_chains)])  # (n_chains, k)
            
            # Create a 4D tensor to handle all chain-specific ensembles at once
            # complementary_states[indices] gives us (n_chains, k, flat_dim)
            selected_ensembles = complementary_states[indices]  # (n_chains, k, flat_dim)
            
            # Compute chain-specific centered and normalized ensembles
            ensemble_means = np.mean(selected_ensembles, axis=1, keepdims=True)  # (n_chains, 1, flat_dim)
            centered_ensembles = (selected_ensembles - ensemble_means) / np.sqrt(k)  # (n_chains, k, flat_dim)
            
            # Generate noise for each chain in k-dimensional subspace
            z = np.random.randn(n_chains, k)  # (n_chains, k)
            
            # Vectorized Langevin proposal using einsum (similar to Hamiltonian approach)
            # Drift term: -h * B_S * B_S^T * ∇V 
            # First: B_S^T * ∇V = einsum('ndk,nd->nk') gives (n_chains, k)
            # Then: B_S * (B_S^T * ∇V) = einsum('nkd,nk->nd') gives (n_chains, flat_dim)
            bt_grad = np.einsum('nkd,nd->nk', centered_ensembles, gradients)  # (n_chains, k)
            drift_term = -current_h * np.einsum('nkd,nk->nd', centered_ensembles, bt_grad)  # (n_chains, flat_dim)
            
            # Noise term: √(2h) * B_S * z where B_S * z is computed via einsum  
            noise_term = np.sqrt(2 * current_h) * np.einsum('nkd,nk->nd', centered_ensembles, z)  # (n_chains, flat_dim)
            
            proposed_states = current_states + drift_term + noise_term
            
            # Compute proposed gradients
            proposed_reshaped = proposed_states.reshape(n_chains, *orig_dim)
            proposed_gradients = gradient_func(proposed_reshaped).reshape(n_chains, -1)
            proposed_gradients = np.nan_to_num(proposed_gradients, nan=0.0)
            
            # Vectorized proposal probability computation using einsum
            bt_grad_current = np.einsum('nkd,nd->nk', centered_ensembles, gradients)  # (n_chains, k)
            bt_grad_proposed = np.einsum('nkd,nd->nk', centered_ensembles, proposed_gradients)  # (n_chains, k)
            
            mean_forward = current_states - current_h * np.einsum('nkd,nk->nd', centered_ensembles, bt_grad_current)
            mean_reverse = proposed_states - current_h * np.einsum('nkd,nk->nd', centered_ensembles, bt_grad_proposed)
            
            residual_forward = proposed_states - mean_forward  # (n_chains, flat_dim)
            residual_reverse = current_states - mean_reverse   # (n_chains, flat_dim)
            
            # Vectorized quadratic form computation using Gram matrix approach (like EKM)
            # Instead of inverting B_S * B_S^T, work in k-dimensional space with B_S^T * B_S
            # For covariance 2h * B_S * B_S^T, the quadratic form becomes:
            # residual^T * (B_S * B_S^T)^(-1) * residual = alpha^T * alpha where B_S^T * B_S * alpha = B_S^T * residual
            
            # Compute Gram matrices: B_S^T @ B_S for all chains (k x k matrices)
            # centered_ensembles has shape (n_chains, k, flat_dim)
            # For B_S^T @ B_S, we contract over the flat_dim dimension
            gram_matrices = np.einsum('nkd,njd->nkj', centered_ensembles, centered_ensembles)  # (n_chains, k, k)
            reg_gram = gram_matrices + 1e-12 * np.eye(k)  # Broadcasting: (n_chains, k, k) + (k, k)
            
            try:
                # Project residuals to k-dimensional space: B_S^T @ residual
                projected_forward = np.einsum('nkd,nd->nk', centered_ensembles, residual_forward)  # (n_chains, k)
                projected_reverse = np.einsum('nkd,nd->nk', centered_ensembles, residual_reverse)  # (n_chains, k)
                
                # Solve in k-dimensional space: (B_S^T @ B_S) @ alpha = B_S^T @ residual
                alpha_forward = np.linalg.solve(reg_gram, projected_forward[..., None])[..., 0]  # (n_chains, k)
                alpha_reverse = np.linalg.solve(reg_gram, projected_reverse[..., None])[..., 0]  # (n_chains, k)
                
                # Quadratic forms: alpha^T @ alpha
                log_q_forward = -np.sum(alpha_forward**2, axis=1) / (4 * current_h)
                log_q_reverse = -np.sum(alpha_reverse**2, axis=1) / (4 * current_h)
                
            except np.linalg.LinAlgError:
                # Fallback: use pseudoinverse of Gram matrices
                inv_gram = np.linalg.pinv(reg_gram)  # (n_chains, k, k)
                
                projected_forward = np.einsum('nkd,nd->nk', centered_ensembles, residual_forward)
                projected_reverse = np.einsum('nkd,nd->nk', centered_ensembles, residual_reverse)
                
                alpha_forward = np.einsum('nkj,nj->nk', inv_gram, projected_forward)
                alpha_reverse = np.einsum('nkj,nj->nk', inv_gram, projected_reverse)
                
                log_q_forward = -np.sum(alpha_forward**2, axis=1) / (4 * current_h)
                log_q_reverse = -np.sum(alpha_reverse**2, axis=1) / (4 * current_h)
        
        return proposed_states, log_q_forward, log_q_reverse
    
    # Main sampling loop
    for i in range(total_iterations):
        is_warmup = i < n_warmup
        current_h = h if not is_warmup else np.exp(log_h)
        
        # Store current state from all chains (only during sampling phase)
        if not is_warmup and (i - n_warmup) % n_thin == 0 and sample_idx < n_samples:
            samples[:, sample_idx] = states
            sample_idx += 1
        
        # First group update using ensemble from group 2
        current_q1 = states[group1].copy()
        current_q1_reshaped = current_q1.reshape(n_chains_per_group, *orig_dim)
        current_U1 = potential_func(current_q1_reshaped)
        
        # Compute gradient
        grad1 = gradient_func(current_q1_reshaped).reshape(n_chains_per_group, -1)
        grad1 = np.nan_to_num(grad1, nan=0.0)
        
        # Vectorized group update
        proposed_q1, log_q_forward1, log_q_reverse1 = vectorized_group_update(
            current_q1, grad1, states[group2])
        
        # Compute proposed energy
        proposed_q1_reshaped = proposed_q1.reshape(n_chains_per_group, *orig_dim)
        proposed_U1 = potential_func(proposed_q1_reshaped)
        
        # Metropolis-Hastings ratio
        dU1 = proposed_U1 - current_U1
        log_ratio = -dU1 + log_q_reverse1 - log_q_forward1
        
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
        current_q2 = states[group2].copy()
        current_q2_reshaped = current_q2.reshape(n_chains_per_group, *orig_dim)
        current_U2 = potential_func(current_q2_reshaped)
        
        grad2 = gradient_func(current_q2_reshaped).reshape(n_chains_per_group, -1)
        grad2 = np.nan_to_num(grad2, nan=0.0)
        
        # Vectorized group update
        proposed_q2, log_q_forward2, log_q_reverse2 = vectorized_group_update(
            current_q2, grad2, states[group1])
        
        # Compute proposed energy
        proposed_q2_reshaped = proposed_q2.reshape(n_chains_per_group, *orig_dim)
        proposed_U2 = potential_func(proposed_q2_reshaped)
        
        # Metropolis-Hastings ratio
        dU2 = proposed_U2 - current_U2
        log_ratio = -dU2 + log_q_reverse2 - log_q_forward2
        
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
        
    
    # Initial state
    initial = np.zeros(dim)
    
    # Dictionary to store results
    results = {}
    
    # Define samplers to benchmark with burn-in
    total_samples = n_samples
    
    
    # Define samplers to benchmark - adjust parameters for high-dimensional case
    samplers = {
        "Langevin Walk Move Ensemble": lambda: langevin_walk_move_k_subset(gradient_func=gradient, potential_func=potential, k=2, initial=initial, n_samples=total_samples, n_warmup=burn_in, n_chains_per_group=dim, h=1.362/(dim**(1/2)), target_accept=0.57, n_thin=n_thin),
        # "Langevin Walk Move Full": lambda: langevin_walk_move_ensemble_dual_avg(gradient_func=gradient, potential_func=potential, initial=initial, n_samples=total_samples, n_warmup=burn_in, n_chains_per_group=dim, h=1.362/(dim**(1/2)), target_accept=0.57, n_thin=n_thin),
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
burn_in = 2*10**3
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
