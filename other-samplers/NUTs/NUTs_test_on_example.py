###### Not tested
###### Goal is to identify examples that NUTS are necessary so motivate the application of PEANUTS

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from collections import defaultdict
import time

# Set random seed for reproducibility
np.random.seed(42)

# Generate unbalanced hierarchical data
def generate_hierarchical_data():
    """Generate hierarchical logistic regression data with unbalanced groups"""
    
    # True parameters
    mu_alpha_true = 0.5
    sigma_alpha_true = 0.8
    beta_true = 1.2
    
    # Group configurations: (n_customers, base_conversion_rate)
    groups = {
        'Channel_A': (1000, 0.45),  # Large, well-identified
        'Channel_B': (1000, 0.38),  # Large, well-identified  
        'Channel_C': (50, 0.24),    # Small, poorly-identified
        'Channel_D': (20, 0.75),    # Small, poorly-identified
        'Channel_E': (800, 0.52),   # Large, well-identified
    }
    
    data = []
    group_effects_true = {}
    
    for group_name, (n_customers, base_rate) in groups.items():
        # True group effect (convert base_rate to logit scale roughly)
        alpha_j_true = np.log(base_rate / (1 - base_rate)) + np.random.normal(0, sigma_alpha_true)
        group_effects_true[group_name] = alpha_j_true
        
        # Generate customer features and outcomes
        x = np.random.normal(0, 1, n_customers)  # customer feature
        logits = alpha_j_true + beta_true * x
        probs = 1 / (1 + np.exp(-logits))
        y = np.random.binomial(1, probs)
        
        for i in range(n_customers):
            data.append({
                'group': group_name,
                'x': x[i], 
                'y': y[i],
                'group_idx': list(groups.keys()).index(group_name)
            })
    
    print("Data Summary:")
    for group_name, (n_customers, _) in groups.items():
        group_data = [d for d in data if d['group'] == group_name]
        conversion_rate = np.mean([d['y'] for d in group_data])
        print(f"{group_name}: {n_customers:4d} customers, {conversion_rate:.3f} conversion rate")
    
    return data, group_effects_true, (mu_alpha_true, sigma_alpha_true, beta_true)

def log_posterior(params, data, n_groups):
    """Log posterior for hierarchical logistic regression"""
    mu_alpha, log_sigma_alpha, beta = params[:3]
    alphas = params[3:3+n_groups]
    
    sigma_alpha = np.exp(log_sigma_alpha)
    
    # Priors
    log_prior = (stats.norm.logpdf(mu_alpha, 0, 1) + 
                stats.norm.logpdf(log_sigma_alpha, 0, 1) +  # log-normal prior for sigma
                stats.norm.logpdf(beta, 0, 1) +
                np.sum(stats.norm.logpdf(alphas, mu_alpha, sigma_alpha)))
    
    # Likelihood
    log_likelihood = 0
    for d in data:
        logit = alphas[d['group_idx']] + beta * d['x']
        prob = 1 / (1 + np.exp(-logit))
        prob = np.clip(prob, 1e-15, 1-1e-15)  # numerical stability
        log_likelihood += d['y'] * np.log(prob) + (1-d['y']) * np.log(1-prob)
    
    return log_prior + log_likelihood

def gradient_log_posterior(params, data, n_groups):
    """Gradient of log posterior"""
    mu_alpha, log_sigma_alpha, beta = params[:3]
    alphas = params[3:3+n_groups]
    sigma_alpha = np.exp(log_sigma_alpha)
    
    grad = np.zeros_like(params)
    
    # Prior gradients
    grad[0] = -mu_alpha  # d/d_mu_alpha
    grad[1] = -log_sigma_alpha  # d/d_log_sigma_alpha (from log-normal prior)
    grad[2] = -beta  # d/d_beta
    
    # Hierarchical prior gradients for alphas
    alpha_deviations = alphas - mu_alpha
    grad[0] += np.sum(alpha_deviations) / sigma_alpha**2  # contribution to mu_alpha
    grad[1] += -n_groups + np.sum(alpha_deviations**2) / sigma_alpha**2  # contribution to log_sigma_alpha
    grad[3:3+n_groups] = -alpha_deviations / sigma_alpha**2  # d/d_alphas
    
    # Likelihood gradients
    group_grad_alpha = np.zeros(n_groups)
    grad_beta = 0
    
    for d in data:
        logit = alphas[d['group_idx']] + beta * d['x']
        prob = 1 / (1 + np.exp(-logit))
        residual = d['y'] - prob
        
        group_grad_alpha[d['group_idx']] += residual
        grad_beta += residual * d['x']
    
    grad[2] += grad_beta
    grad[3:3+n_groups] += group_grad_alpha
    
    return grad

def leapfrog(params, momentum, stepsize, data, n_groups):
    """Single leapfrog step"""
    # Half step for momentum
    grad = gradient_log_posterior(params, data, n_groups)
    momentum = momentum + 0.5 * stepsize * grad
    
    # Full step for position
    params = params + stepsize * momentum
    
    # Half step for momentum
    grad = gradient_log_posterior(params, data, n_groups)
    momentum = momentum + 0.5 * stepsize * grad
    
    return params, momentum

def standard_hmc(data, n_groups, n_samples=1000, stepsize=0.01, n_steps=20):
    """Standard HMC with fixed trajectory length"""
    n_params = 3 + n_groups
    samples = []
    
    # Initialize
    params = np.random.normal(0, 0.1, n_params)
    
    n_accept = 0
    
    for i in range(n_samples):
        # Generate momentum
        momentum = np.random.normal(0, 1, n_params)
        
        # Store initial state
        params_init = params.copy()
        momentum_init = momentum.copy()
        
        # Initial energy
        kinetic_init = 0.5 * np.sum(momentum_init**2)
        potential_init = -log_posterior(params_init, data, n_groups)
        energy_init = kinetic_init + potential_init
        
        # Simulate Hamiltonian dynamics
        for _ in range(n_steps):
            params, momentum = leapfrog(params, momentum, stepsize, data, n_groups)
        
        # Final energy
        kinetic_final = 0.5 * np.sum(momentum**2)
        potential_final = -log_posterior(params, data, n_groups)
        energy_final = kinetic_final + potential_final
        
        # Accept/reject
        accept_prob = min(1, np.exp(energy_init - energy_final))
        
        if np.random.rand() < accept_prob:
            n_accept += 1
        else:
            params = params_init
        
        samples.append(params.copy())
        
        if (i+1) % 200 == 0:
            print(f"HMC: {i+1}/{n_samples}, accept rate: {n_accept/(i+1):.3f}")
    
    return np.array(samples), n_accept/n_samples

def nuts_sampler(data, n_groups, n_samples=1000, stepsize=0.01, max_treedepth=10):
    """Simple NUTS implementation (simplified version)"""
    n_params = 3 + n_groups
    samples = []
    
    # Initialize
    params = np.random.normal(0, 0.1, n_params)
    
    def build_tree(params, momentum, u, direction, depth, stepsize):
        """Build tree for NUTS (simplified)"""
        if depth == 0:
            # Base case: single leapfrog step
            if direction == 1:
                params_new, momentum_new = leapfrog(params, momentum, stepsize, data, n_groups)
            else:
                params_new, momentum_new = leapfrog(params, momentum, -stepsize, data, n_groups)
            
            # Check if valid
            log_prob = log_posterior(params_new, data, n_groups)
            kinetic = 0.5 * np.sum(momentum_new**2)
            
            valid = (u <= np.exp(log_prob - kinetic))
            return params_new, momentum_new, params_new, momentum_new, params_new, int(valid), 1
        
        else:
            # Recursion
            # First half of tree
            params_minus, momentum_minus, params_plus, momentum_plus, params_prime, n_valid, n_steps = \
                build_tree(params, momentum, u, direction, depth-1, stepsize)
            
            if n_valid > 0:  # No U-turn yet
                if direction == 1:
                    params_plus, momentum_plus, _, _, params_double_prime, n_valid_prime, n_steps_prime = \
                        build_tree(params_plus, momentum_plus, u, direction, depth-1, stepsize)
                else:
                    _, _, params_minus, momentum_minus, params_double_prime, n_valid_prime, n_steps_prime = \
                        build_tree(params_minus, momentum_minus, u, direction, depth-1, stepsize)
                
                # Choose which candidate to return
                if np.random.rand() < n_valid_prime / max(n_valid + n_valid_prime, 1):
                    params_prime = params_double_prime
                
                # Check U-turn condition (simplified)
                delta_minus = params_minus - params
                delta_plus = params_plus - params
                no_uturn = (np.dot(delta_minus, momentum_minus) >= 0 and 
                           np.dot(delta_plus, momentum_plus) >= 0)
                
                n_valid = n_valid + n_valid_prime if no_uturn else 0
                n_steps = n_steps + n_steps_prime
            
            return params_minus, momentum_minus, params_plus, momentum_plus, params_prime, n_valid, n_steps
    
    total_steps = 0
    
    for i in range(n_samples):
        # Generate momentum
        momentum = np.random.normal(0, 1, n_params)
        
        # Slice variable
        log_prob = log_posterior(params, data, n_groups)
        kinetic = 0.5 * np.sum(momentum**2)
        u = np.random.uniform(0, np.exp(log_prob - kinetic))
        
        # Initialize tree
        params_minus = params_plus = params.copy()
        momentum_minus = momentum_plus = momentum.copy()
        
        depth = 0
        n_valid = 1
        params_new = params.copy()
        
        # Build tree
        while n_valid > 0 and depth < max_treedepth:
            # Choose direction
            direction = 1 if np.random.rand() < 0.5 else -1
            
            if direction == 1:
                _, _, params_plus, momentum_plus, params_candidate, n_valid_new, steps = \
                    build_tree(params_plus, momentum_plus, u, direction, depth, stepsize)
            else:
                params_minus, momentum_minus, _, _, params_candidate, n_valid_new, steps = \
                    build_tree(params_minus, momentum_minus, u, direction, depth, stepsize)
            
            total_steps += steps
            
            if n_valid_new > 0:
                if np.random.rand() < n_valid_new / (n_valid + n_valid_new):
                    params_new = params_candidate
            
            n_valid += n_valid_new
            depth += 1
        
        params = params_new
        samples.append(params.copy())
        
        if (i+1) % 200 == 0:
            avg_steps = total_steps / (i+1)
            print(f"NUTS: {i+1}/{n_samples}, avg steps per sample: {avg_steps:.1f}")
    
    return np.array(samples), total_steps

def analyze_results(hmc_samples, nuts_samples, true_params, data):
    """Compare HMC vs NUTS results"""
    mu_alpha_true, sigma_alpha_true, beta_true = true_params
    
    # Extract parameters
    hmc_mu = hmc_samples[:, 0]
    hmc_beta = hmc_samples[:, 2]
    nuts_mu = nuts_samples[:, 0] 
    nuts_beta = nuts_samples[:, 2]
    
    # Effective sample size (simple autocorrelation estimate)
    def eff_sample_size(x, max_lag=100):
        n = len(x)
        x_centered = x - np.mean(x)
        autocorr = np.correlate(x_centered, x_centered, mode='full')
        autocorr = autocorr[n-1:n-1+max_lag]
        autocorr = autocorr / autocorr[0]
        
        # Find first negative autocorrelation
        first_negative = np.where(autocorr < 0)[0]
        if len(first_negative) > 0:
            cutoff = first_negative[0]
        else:
            cutoff = max_lag
        
        # Sum of autocorrelations
        tau = 1 + 2 * np.sum(autocorr[1:cutoff])
        return n / tau
    
    hmc_ess_mu = eff_sample_size(hmc_mu)
    nuts_ess_mu = eff_sample_size(nuts_mu)
    hmc_ess_beta = eff_sample_size(hmc_beta)
    nuts_ess_beta = eff_sample_size(nuts_beta)
    
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    print(f"Parameter μ_α (true: {mu_alpha_true:.3f})")
    print(f"  HMC:  mean={np.mean(hmc_mu):.3f}, std={np.std(hmc_mu):.3f}, ESS={hmc_ess_mu:.1f}")
    print(f"  NUTS: mean={np.mean(nuts_mu):.3f}, std={np.std(nuts_mu):.3f}, ESS={nuts_ess_mu:.1f}")
    
    print(f"\nParameter β (true: {beta_true:.3f})")
    print(f"  HMC:  mean={np.mean(hmc_beta):.3f}, std={np.std(hmc_beta):.3f}, ESS={hmc_ess_beta:.1f}")
    print(f"  NUTS: mean={np.mean(nuts_beta):.3f}, std={np.std(nuts_beta):.3f}, ESS={nuts_ess_beta:.1f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Trace plots
    axes[0,0].plot(hmc_mu, alpha=0.7, label='HMC')
    axes[0,0].axhline(mu_alpha_true, color='red', linestyle='--', label='True')
    axes[0,0].set_title('μ_α trace')
    axes[0,0].legend()
    
    axes[0,1].plot(nuts_mu, alpha=0.7, label='NUTS', color='orange')
    axes[0,1].axhline(mu_alpha_true, color='red', linestyle='--', label='True')
    axes[0,1].set_title('μ_α trace (NUTS)')
    axes[0,1].legend()
    
    # Posterior distributions
    axes[1,0].hist(hmc_mu, bins=30, alpha=0.7, density=True, label='HMC')
    axes[1,0].hist(nuts_mu, bins=30, alpha=0.7, density=True, label='NUTS')
    axes[1,0].axvline(mu_alpha_true, color='red', linestyle='--', label='True')
    axes[1,0].set_title('μ_α posterior')
    axes[1,0].legend()
    
    axes[1,1].hist(hmc_beta, bins=30, alpha=0.7, density=True, label='HMC')
    axes[1,1].hist(nuts_beta, bins=30, alpha=0.7, density=True, label='NUTS')
    axes[1,1].axvline(beta_true, color='red', linestyle='--', label='True')
    axes[1,1].set_title('β posterior')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'hmc_ess': (hmc_ess_mu + hmc_ess_beta) / 2,
        'nuts_ess': (nuts_ess_mu + nuts_ess_beta) / 2
    }

if __name__ == "__main__":
    # Generate data
    print("Generating hierarchical data...")
    data, true_group_effects, true_params = generate_hierarchical_data()
    n_groups = len(set(d['group'] for d in data))
    
    print(f"\nRunning samplers on {len(data)} observations, {n_groups} groups...")
    
    # Run HMC
    print("\nRunning standard HMC...")
    start_time = time.time()
    hmc_samples, hmc_accept_rate = standard_hmc(data, n_groups, n_samples=1000, 
                                               stepsize=0.005, n_steps=25)
    hmc_time = time.time() - start_time
    print(f"HMC completed in {hmc_time:.1f}s, accept rate: {hmc_accept_rate:.3f}")
    
    # Run NUTS
    print("\nRunning NUTS...")
    start_time = time.time()
    nuts_samples, total_nuts_steps = nuts_sampler(data, n_groups, n_samples=1000, 
                                                 stepsize=0.005, max_treedepth=8)
    nuts_time = time.time() - start_time
    avg_nuts_steps = total_nuts_steps / len(nuts_samples)
    print(f"NUTS completed in {nuts_time:.1f}s, avg {avg_nuts_steps:.1f} steps/sample")
    
    # Analyze results
    results = analyze_results(hmc_samples, nuts_samples, true_params, data)
    
    print(f"\nEfficiency comparison:")
    print(f"HMC:  {results['hmc_ess']:.1f} effective samples")
    print(f"NUTS: {results['nuts_ess']:.1f} effective samples")
    print(f"NUTS advantage: {results['nuts_ess']/results['hmc_ess']:.1f}x effective sample size")
    print(f"NUTS cost: {avg_nuts_steps/25:.1f}x more gradient evaluations per sample")