import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize, differential_evolution
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

#from fixed_point_v2 import (fourier_multiplier, double_well_potential, energy_value, N_eps_v2, fixpoint_corrected, initialize_field_random)
# same functions as in fixed_point_v2

class MagneticDomainOptimizer:
    """Complete parameter optimization system for magnetic domain simulation"""
    
    def __init__(self, target_image, simulation_params=None):
        """
        Initialize the optimizer
        
        Args:
            target_image: Experimental image (numpy array)
            simulation_params: Dict with simulation settings
        """
        self.target_image = self.preprocess_image(target_image)
        
        # Default simulation parameters
        self.sim_params = {
            'N': 256,
            'dt': 1/100,
            'max_it': 500,  # Reduced for optimization
            'max_it_fixpoint': 50,
            'tol': 1e-8,
            'gridsize': 1,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        if simulation_params:
            self.sim_params.update(simulation_params)
        
        # Parameter bounds for optimization
        self.param_bounds = {
            'gamma': (1e-6, 1e-2),
            'epsilon': (0.01, 1.0),
            'th': (0.01, 1.0)
        }
        
        # Loss function weights
        self.loss_weights = {
            'mse': 1.0,
            'ssim': 2.0,
            'gradient': 1.5,
            'physics': 0.5
        }
        
        # Optimization history
        self.optimization_history = []
        
    def preprocess_image(self, image):
        """Preprocess experimental image"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to simulation grid size
        image = cv2.resize(image, (256, 256))
        
        # Normalize
        image = image.astype(np.float64)
        image = (image - image.mean()) / image.std()
        
        # Optional: denoise
        image = gaussian_filter(image, sigma=0.5)
        
        return image
    
    def run_simulation(self, params):
        """
        Run magnetic domain simulation with given parameters
        
        Args:
            params: [gamma, epsilon, th, c0]
        
        Returns:
            Final simulation image as numpy array
        """
        gamma, epsilon, th = params

        c0 = 9/32
        
        try:
            # Setup device and data types
            dtype_real = torch.float64
            dtype_complex = torch.complex128
            device = self.sim_params['device']
            N = self.sim_params['N']
            
            # Grid setup
            k = torch.fft.fftfreq(N, d=1/N, device=device)
            xi, eta = torch.meshgrid(k, k, indexing='ij')
            modk2 = (xi ** 2 + eta ** 2).to(dtype_real)
            modk = torch.sqrt(modk2)
            
            # Fourier multiplier - using your dipolar version
            def fourier_multiplier_dipolar(k_mag, thickness):
                k_safe = torch.where(k_mag < 1e-12, torch.tensor(1e-12, dtype=dtype_real, device=device), k_mag)
                sigma = 1.0 - torch.exp(-k_safe * thickness)
                sigma = torch.where(k_mag < 1e-12, torch.tensor(0.0, dtype=dtype_real, device=device), sigma)
                return sigma
            
            # Linear operator
            L = (2 * np.pi) ** 2 * gamma * epsilon * modk2 + fourier_multiplier_dipolar(modk, th)
            
            # Initialize field
            u0 = self.initialize_field(N, device, dtype_real)
            
            # Fixed point iteration functions
            def double_well_potential(u, c0_val):
                u2 = 1 - torch.abs(u) ** 2
                return c0_val * (u2 ** 2)
            
            def N_eps_v2(u, eps, gam, c0_val):
                return gam * c0_val / eps * u * (1 - torch.abs(u)**2)
            
            def fixpoint_iteration(U_0, L_op, dt, eps, gam, max_iter, tolerance, c0_val):
                _ones = torch.ones_like(L_op)
                G_m = (_ones - dt / 2 * L_op)
                G_p = (_ones + dt / 2 * L_op)
                G_p = torch.where(torch.abs(G_p) < 1e-12, torch.tensor(1e-12, dtype=G_p.dtype, device=device), G_p)
                
                CT = torch.fft.ifft2(G_m / G_p * torch.fft.fft2(U_0))
                if not torch.is_complex(U_0):
                    CT = CT.real
                
                U_n = U_0.clone()
                
                for ii in range(max_iter):
                    non_linear = N_eps_v2(U_n, eps, gam, c0_val)
                    U_np1 = torch.fft.ifft2(torch.fft.fft2(dt * non_linear) / G_p)
                    if not torch.is_complex(U_0):
                        U_np1 = U_np1.real
                    U_np1 = U_np1 + CT
                    
                    error = torch.max(torch.abs(U_np1 - U_n)).item()
                    U_n = U_np1.clone()
                    
                    if error < tolerance:
                        return U_n, True
                
                return U_n, False
            
            # Run simulation
            u_n = u0.clone()
            dt = self.sim_params['dt']
            
            for iteration in range(self.sim_params['max_it']):
                u_n, converged = fixpoint_iteration(
                    u_n, L, dt, epsilon, gamma, 
                    self.sim_params['max_it_fixpoint'], 
                    self.sim_params['tol'], c0
                )
                
                if not converged:
                    dt = dt / 2
                    if dt < 1e-10:
                        break
                
                # Early stopping for optimization
                if iteration > 50 and iteration % 25 == 0:
                    break
            
            # Convert to numpy
            if torch.is_complex(u_n):
                result = torch.abs(u_n).cpu().numpy()
            else:
                result = u_n.cpu().numpy()
            
            return result
            
        except Exception as e:
            print(f"Simulation failed with params {params}: {e}")
            # Return a high-loss dummy image
            return np.random.randn(self.sim_params['N'], self.sim_params['N'])
    
    def initialize_field(self, N, device, dtype):
        """Initialize the simulation field"""
        # Start with small random perturbations
        u0 = torch.ones((N, N), dtype=dtype, device=device)
        u0 += 0.1 * (torch.rand(N, N, dtype=dtype, device=device) - 0.5)
        
        # Add some smooth structure
        u0_np = u0.cpu().numpy()
        u0_smooth = gaussian_filter(u0_np, sigma=2.0, mode='wrap')
        u0 = torch.from_numpy(u0_smooth).to(device)
        
        return u0
    
    def calculate_loss(self, params):
        """
        Calculate comprehensive loss function
        
        Args:
            params: [gamma, epsilon, th, c0]
        
        Returns:
            Total loss value
        """
        # Run simulation
        sim_image = self.run_simulation(params)
        
        # Normalize both images
        target_norm = (self.target_image - self.target_image.mean()) / self.target_image.std()
        sim_norm = (sim_image - sim_image.mean()) / sim_image.std()
        
        # Component losses
        losses = {}
        
        # 1. MSE Loss
        losses['mse'] = np.mean((target_norm - sim_norm)**2)
        
        # 2. SSIM Loss (1 - SSIM for minimization)
        try:
            ssim_val = ssim(target_norm, sim_norm, data_range=target_norm.max() - target_norm.min())
            losses['ssim'] = 1 - ssim_val
        except:
            losses['ssim'] = 1.0
        
        # 3. Gradient Loss
        target_grad = np.abs(sobel(target_norm))
        sim_grad = np.abs(sobel(sim_norm))
        losses['gradient'] = np.mean((target_grad - sim_grad)**2)
        
        # 4. Physics Constraints
        gamma, epsilon, th = params
        physics_penalty = 0.0
        
        # Parameter bounds penalty
        for i, (param, (min_val, max_val)) in enumerate(zip(params, self.param_bounds.values())):
            if param < min_val:
                physics_penalty += 100 * (min_val - param)**2
            elif param > max_val:
                physics_penalty += 100 * (param - max_val)**2
        
        # Physical relationships
        if epsilon > 0.5 and gamma > 1e-3:  # Avoid unrealistic combinations
            physics_penalty += 10 * (epsilon * gamma - 5e-4)**2
        
        losses['physics'] = physics_penalty
        
        # Total weighted loss
        total_loss = sum(self.loss_weights[key] * losses[key] for key in losses)
        
        # Store in history
        self.optimization_history.append({
            'params': params.copy(),
            'total_loss': total_loss,
            'component_losses': losses.copy()
        })
        
        print(f"Params: γ={gamma:.2e}, ε={epsilon:.3f}, th={th:.3f} | "
              f"Loss: {total_loss:.4f} (MSE:{losses['mse']:.3f}, SSIM:{losses['ssim']:.3f}, "
              f"Grad:{losses['gradient']:.3f}, Phys:{losses['physics']:.3f})")
        
        return total_loss
    
    def optimize_scipy(self, method='L-BFGS-B', max_iterations=100):
        """Optimize using scipy methods"""
        print(f"Starting optimization with {method}...")
        
        # Initial guess
        x0 = [1e-3, 0.1, 0.1, 9/32]
        
        # Bounds
        bounds = list(self.param_bounds.values())
        
        # Optimize
        start_time = time.time()
        result = minimize(
            self.calculate_loss, x0, method=method, bounds=bounds,
            options={'maxiter': max_iterations, 'disp': True}
        )
        
        optimization_time = time.time() - start_time
        
        print(f"\nOptimization completed in {optimization_time:.1f} seconds")
        print(f"Success: {result.success}")
        print(f"Final loss: {result.fun:.6f}")
        print(f"Optimal parameters:")
        print(f"  gamma = {result.x[0]:.2e}")
        print(f"  epsilon = {result.x[1]:.6f}")
        print(f"  th = {result.x[2]:.6f}")
        #print(f"  c0 = {result.x[3]:.6f}")
        
        return result
    
    def optimize_differential_evolution(self, max_iterations=50):
        """Optimize using differential evolution (more robust for global optimization)"""
        print("Starting differential evolution optimization...")
        
        # Bounds
        bounds = list(self.param_bounds.values())
        
        # Optimize
        start_time = time.time()
        result = differential_evolution(
            self.calculate_loss, bounds, maxiter=max_iterations,
            popsize=10, seed=42, disp=True, polish=True
        )
        
        optimization_time = time.time() - start_time
        
        print(f"\nOptimization completed in {optimization_time:.1f} seconds")
        print(f"Success: {result.success}")
        print(f"Final loss: {result.fun:.6f}")
        print(f"Optimal parameters:")
        print(f"  gamma = {result.x[0]:.2e}")
        print(f"  epsilon = {result.x[1]:.6f}")
        print(f"  th = {result.x[2]:.6f}")
        # print(f"  c0 = {result.x[3]:.6f}")
        
        return result
    
    def plot_optimization_progress(self):
        """Plot optimization history"""
        if not self.optimization_history:
            print("No optimization history to plot")
            return
        
        history = self.optimization_history
        iterations = range(len(history))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Total loss
        total_losses = [h['total_loss'] for h in history]
        axes[0,0].plot(iterations, total_losses)
        axes[0,0].set_title('Total Loss')
        axes[0,0].set_yscale('log')
        axes[0,0].grid(True)
        
        # Component losses
        loss_types = ['mse', 'ssim', 'gradient', 'physics']
        for i, loss_type in enumerate(loss_types):
            row, col = (i+1)//3, (i+1)%3
            losses = [h['component_losses'][loss_type] for h in history]
            axes[row,col].plot(iterations, losses)
            axes[row,col].set_title(f'{loss_type.upper()} Loss')
            axes[row,col].grid(True)
        
        # Parameter evolution
        param_names = ['gamma', 'epsilon', 'th']
        axes[1,2].clear()
        for i, param_name in enumerate(param_names):
            params = [h['params'][i] for h in history]
            axes[1,2].plot(iterations, params, label=param_name)
        axes[1,2].set_title('Parameter Evolution')
        axes[1,2].legend()
        axes[1,2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def compare_results(self, optimal_params):
        """Compare simulation result with target image"""
        # Run simulation with optimal parameters
        sim_result = self.run_simulation(optimal_params)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Target image
        im1 = axes[0].imshow(self.target_image, cmap='RdBu', extent=[0,1,0,1])
        axes[0].set_title('Target (Experimental)')
        plt.colorbar(im1, ax=axes[0])
        
        # Simulated image
        im2 = axes[1].imshow(sim_result, cmap='RdBu', extent=[0,1,0,1])
        axes[1].set_title('Simulated (Optimized)')
        plt.colorbar(im2, ax=axes[1])
        
        # Difference
        diff = self.target_image - (sim_result - sim_result.mean()) / sim_result.std()
        im3 = axes[2].imshow(diff, cmap='viridis', extent=[0,1,0,1])
        axes[2].set_title('Difference')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.show()
        
        # Calculate final metrics
        target_norm = (self.target_image - self.target_image.mean()) / self.target_image.std()
        sim_norm = (sim_result - sim_result.mean()) / sim_result.std()
        
        final_mse = np.mean((target_norm - sim_norm)**2)
        final_ssim = ssim(target_norm, sim_norm, data_range=target_norm.max() - target_norm.min())
        
        print(f"\nFinal comparison metrics:")
        print(f"MSE: {final_mse:.6f}")
        print(f"SSIM: {final_ssim:.6f}")
        
        return sim_result
    
    def save_results(self, result, filename='optimization_results.pkl'):
        """Save optimization results"""
        save_data = {
            'optimal_params': result.x,
            'final_loss': result.fun,
            'optimization_history': self.optimization_history,
            'target_image': self.target_image,
            'param_bounds': self.param_bounds,
            'loss_weights': self.loss_weights
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Results saved to {filename}")

# Usage example
def main_optimization_workflow():
    """Complete workflow for parameter optimization"""
    
    # 1. Load experimental image
    # Replace this with your actual experimental image
    print("Loading experimental image...")
    
    # Example: create synthetic "experimental" data for testing
    # In practice, you would load: 
    experimental_image = cv2.imread('input_test.png', 0)
    #experimental_image = np.random.randn(256, 256)
    #experimental_image = gaussian_filter(experimental_image, sigma=3)
    #experimental_image = (experimental_image > 0).astype(float) * 2 - 1  # Binary-like domains
    
    # 2. Initialize optimizer
    print("Initializing optimizer...")
    optimizer = MagneticDomainOptimizer(experimental_image)
    
    # 3. Optimize parameters
    print("Starting parameter optimization...")
    
    # Method 1: Differential Evolution (recommended for global optimization)
    result_de = optimizer.optimize_differential_evolution(max_iterations=30)
    
    # Method 2: Local optimization from DE result
    print("\nRefining with local optimization...")
    optimizer.optimization_history = []  # Reset history for cleaner plotting
    
    # Use DE result as starting point for local optimization
    x0 = result_de.x
    bounds = list(optimizer.param_bounds.values())
    result_local = minimize(
        optimizer.calculate_loss, x0, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 50}
    )
    
    # 4. Analyze results
    print("\nAnalyzing results...")
    optimizer.plot_optimization_progress()
    
    # 5. Compare final result
    print("Comparing final result...")
    final_simulation = optimizer.compare_results(result_local.x)
    
    # 6. Save results
    optimizer.save_results(result_local)
    
    return result_local.x, optimizer

if __name__ == "__main__":
    optimal_parameters, optimizer = main_optimization_workflow()
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Use these parameters in your simulation:")
    print(f"gamma = {optimal_parameters[0]:.2e}")
    print(f"epsilon = {optimal_parameters[1]:.6f}")
    print(f"th = {optimal_parameters[2]:.6f}")
    #print(f"c0 = {optimal_parameters[3]:.6f}")