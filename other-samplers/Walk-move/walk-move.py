def walk_move(log_prob, initial, n_samples, n_walkers=20, n_thin=1, subset_size=None, stepsize=1.0):
    """
    Vectorized implementation of the Ensemble MCMC with walk moves.
    
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
        
    Returns:
    --------
    samples : array
        Samples from all walkers (shape: n_walkers, n_samples, dim)
    acceptance_rates : array
        Acceptance rates for all walkers
    """
    import numpy as np
    
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
    
    # Calculate total iterations needed based on thinning factor
    total_iterations = n_samples * n_thin
    
    # Storage for samples and tracking acceptance
    samples = np.zeros((n_walkers, n_samples, dim))
    accepts = np.zeros(n_walkers)
    
    # Sample index to track where to store thinned samples
    sample_idx = 0
    
    # Main sampling loop
    for i in range(total_iterations):
        # Store current state from all walkers (only every n_thin iterations)
        if i % n_thin == 0 and sample_idx < n_samples:
            samples[:, sample_idx] = walkers
            sample_idx += 1
        
        # Update first half, then second half
        for half in [0, 1]:
            # Set indices for active and complementary walker sets
            active_indices = np.arange(half * half_walkers, (half + 1) * half_walkers)
            comp_indices = np.arange((1 - half) * half_walkers, (2 - half) * half_walkers)
            
            # Get complementary walkers and select subset S of specified size
            comp_indices_full = np.arange((1 - half) * half_walkers, (2 - half) * half_walkers)
            selected_comp_indices = np.random.choice(comp_indices_full, size=subset_size, replace=False)
            comp_walkers = walkers[selected_comp_indices]
            
            # Calculate mean of complementary ensemble subset S
            comp_mean = np.mean(comp_walkers, axis=0)
            
            # Extract active walkers
            active_walkers = walkers[active_indices]
            
            # Vectorized generation of W using equation (11): W = sum(Z_j * (X_j - X_S_mean))
            # Generate all Z_j values at once: shape (half_walkers, subset_size)
            z_values = np.random.randn(half_walkers, subset_size)
            
            # Calculate (X_j - X_S_mean) for all walkers in subset: shape (subset_size, dim)
            centered_walkers = comp_walkers - comp_mean
            
            # Vectorized calculation: W = sum over j of Z_j * (X_j - X_S_mean)
            # Shape: (half_walkers, subset_size) @ (subset_size, dim) = (half_walkers, dim)
            proposal_offsets = stepsize * (z_values @ centered_walkers)
            
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
            
            # Track acceptance for all walkers
            accepts[active_indices[accepted]] += 1
    
    # Return results from all walkers
    acceptance_rates = accepts / total_iterations
    return samples, acceptance_rates