"""
Module for performing small-signal stability assessment in Dynawo.
"""

import dynawo
import numpy as np
import pandas as pd
from scipy import signal


class SmallSignalStabilityAnalyzer:
    """Class to perform small-signal stability analysis in Dynawo power system models."""
    
    def __init__(self, simulation):
        """
        Initialize the small-signal stability analyzer.
        
        Args:
            simulation: A Dynawo simulation object
        """
        self.simulation = simulation
        self.network = simulation.get_network()
        self.results = None
        self.modes = None
    
    def linearize_system(self):
        """
        Linearize the system around the current operating point.
        
        Returns:
            True if linearization was successful, False otherwise
        """
        print("Linearizing system around current operating point...")
        
        try:
            # Create a linearization solver
            linearizer = dynawo.LinearizationSolver(self.simulation)
            
            # Set solver options
            linearizer.set_tolerance(1e-6)
            
            # Run the linearization
            success = linearizer.solve()
            
            if success:
                print("System linearized successfully")
                self.state_matrix = linearizer.get_state_matrix()
                return True
            else:
                print("Linearization failed")
                return False
                
        except Exception as e:
            print(f"Error during linearization: {str(e)}")
            return False
    
    def compute_eigenvalues(self):
        """
        Compute eigenvalues of the state matrix.
        
        Returns:
            NumPy array of complex eigenvalues
        """
        print("Computing eigenvalues...")
        
        # Linearize system if not already done
        if not hasattr(self, 'state_matrix'):
            success = self.linearize_system()
            if not success:
                print("Cannot compute eigenvalues - linearization failed")
                return None
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(self.state_matrix)
        
        # Store results
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        print(f"Computed {len(eigenvalues)} eigenvalues")
        return eigenvalues
    
    def identify_modes(self, freq_min=0.25, freq_max=1.0):
        """
        Identify electromechanical oscillation modes.
        
        Args:
            freq_min: Minimum frequency of interest (Hz)
            freq_max: Maximum frequency of interest (Hz)
            
        Returns:
            DataFrame containing the identified modes
        """
        # Compute eigenvalues if not already done
        if not hasattr(self, 'eigenvalues'):
            self.compute_eigenvalues()
            if self.eigenvalues is None:
                return None
        
        modes = []
        
        # Convert eigenvalues to damping and frequency
        for i, eig in enumerate(self.eigenvalues):
            if eig.imag != 0:  # Only consider oscillatory modes
                freq = abs(eig.imag) / (2 * np.pi)  # Frequency in Hz
                damping = -eig.real / np.sqrt(eig.real**2 + eig.imag**2)
                
                # Check if the mode is in the frequency range of interest
                if freq_min <= freq <= freq_max:
                    modes.append({
                        'mode_idx': i,
                        'eigenvalue_real': eig.real,
                        'eigenvalue_imag': eig.imag,
                        'frequency_hz': freq,
                        'damping_ratio': damping,
                        'time_constant': -1/eig.real if eig.real < 0 else float('inf')
                    })
        
        # Create DataFrame and sort by damping ratio
        self.modes = pd.DataFrame(modes).sort_values(by='damping_ratio')
        
        print(f"Identified {len(self.modes)} electromechanical oscillation modes")
        return self.modes
    
    def check_damping_criterion(self, min_damping=0.03):
        """
        Check if all modes meet the minimum damping criterion.
        
        Args:
            min_damping: Minimum acceptable damping ratio
            
        Returns:
            DataFrame containing poorly damped modes
        """
        # Identify modes if not already done
        if self.modes is None:
            self.identify_modes()
            if self.modes is None:
                return None
        
        # Filter modes that don't meet the damping criterion
        poorly_damped = self.modes[self.modes['damping_ratio'] < min_damping]
        
        if len(poorly_damped) > 0:
            print(f"Found {len(poorly_damped)} poorly damped modes (damping < {min_damping})")
        else:
            print(f"All modes have damping ratio >= {min_damping}")
        
        return poorly_damped
    
    def get_participation_factors(self):
        """
        Calculate participation factors for each state in each mode.
        
        Returns:
            Dictionary mapping mode indices to state participation factors
        """
        # Compute eigenvalues if not already done
        if not hasattr(self, 'eigenvalues') or not hasattr(self, 'eigenvectors'):
            self.compute_eigenvalues()
            if self.eigenvalues is None:
                return None
        
        # Get left eigenvectors
        left_eigenvectors = np.linalg.inv(self.eigenvectors).T
        
        # Calculate participation factors
        participation = {}
        
        for i in range(len(self.eigenvalues)):
            # Calculate participation factors for this mode
            p_factors = np.abs(np.multiply(self.eigenvectors[:, i], left_eigenvectors[:, i]))
            
            # Normalize to sum to 1
            p_factors = p_factors / np.sum(p_factors)
            
            participation[i] = p_factors
        
        return participation
    
    def perform_time_domain_validation(self, duration=20.0, time_step=0.01):
        """
        Perform time-domain simulation to validate the small-signal stability analysis.
        
        Args:
            duration: Simulation duration (seconds)
            time_step: Simulation time step (seconds)
            
        Returns:
            Dictionary containing simulation results
        """
        print("Performing time-domain validation...")
        
        # Set simulation parameters
        self.simulation.set_duration(duration)
        self.simulation.set_time_step(time_step)
        
        # Run the simulation
        result = self.simulation.run()
        
        if result:
            print("Time-domain simulation completed successfully")
            
            # Extract generator rotor angles
            generators = self.network.get_generators()
            gen_angles = {}
            
            for gen in generators:
                gen_id = gen.get_id()
                if hasattr(gen, 'get_rotor_angle'):
                    gen_angles[gen_id] = np.array(gen.get_variable_values('rotor_angle'))
            
            # Perform FFT analysis to identify frequencies
            fft_results = {}
            if len(gen_angles) > 0:
                for gen_id, angles in gen_angles.items():
                    # Detrend data
                    detrended = signal.detrend(angles)
                    
                    # Calculate FFT
                    n = len(detrended)
                    fft = np.fft.fft(detrended)
                    freq = np.fft.fftfreq(n, d=time_step)
                    
                    # Get positive frequencies only
                    positive_idx = np.where(freq > 0)[0]
                    positive_freq = freq[positive_idx]
                    positive_fft = np.abs(fft[positive_idx])
                    
                    # Store results
                    fft_results[gen_id] = {
                        'frequencies': positive_freq,
                        'magnitudes': positive_fft
                    }
            
            return {
                'success': True,
                'generator_angles': gen_angles,
                'fft_results': fft_results
            }
        else:
            print("Time-domain simulation failed")
            return {'success': False}
    
    def assess_small_signal_stability(self, min_damping=0.03, freq_min=0.25, freq_max=1.0):
        """
        Perform a complete small-signal stability assessment.
        
        Args:
            min_damping: Minimum acceptable damping ratio
            freq_min: Minimum frequency of interest (Hz)
            freq_max: Maximum frequency of interest (Hz)
            
        Returns:
            Dictionary containing assessment results
        """
        print("Performing small-signal stability assessment...")
        
        # Linearize the system
        linearized = self.linearize_system()
        if not linearized:
            return {'success': False, 'stable': False, 'message': 'Linearization failed'}
        
        # Compute eigenvalues
        eigenvalues = self.compute_eigenvalues()
        
        # Check if any eigenvalues have positive real parts (unstable)
        unstable_eigenvalues = [ev for ev in eigenvalues if ev.real > 0]
        
        if unstable_eigenvalues:
            print(f"System is small-signal unstable with {len(unstable_eigenvalues)} unstable eigenvalues")
            return {
                'success': True,
                'stable': False,
                'message': f'System has {len(unstable_eigenvalues)} unstable eigenvalues',
                'eigenvalues': eigenvalues,
                'unstable_eigenvalues': unstable_eigenvalues
            }
        
        # Identify oscillatory modes
        self.identify_modes(freq_min=freq_min, freq_max=freq_max)
        
        # Check damping criterion
        poorly_damped = self.check_damping_criterion(min_damping=min_damping)
        
        # Get participation factors for poorly damped modes
        participation = None
        if len(poorly_damped) > 0:
            participation = self.get_participation_factors()
            
            # Filter to only include poorly damped modes
            participation = {mode_idx: factors for mode_idx, factors in participation.items() 
                            if mode_idx in poorly_damped['mode_idx'].values}
        
        # Determine if the system is stable according to the criteria
        is_stable = len(poorly_damped) == 0
        
        # Compile results
        assessment = {
            'success': True,
            'stable': is_stable,
            'eigenvalues': eigenvalues,
            'modes': self.modes,
            'poorly_damped_modes': poorly_damped,
            'participation_factors': participation,
            'min_damping_criterion': min_damping
        }
        
        if is_stable:
            print("System is small-signal stable (all modes meet damping criterion)")
        else:
            print(f"System is small-signal unstable (found {len(poorly_damped)} poorly damped modes)")
        
        return assessment


# Example usage
if __name__ == "__main__":
    # This requires a simulation object to be created first
    # from load_ieee_systems import IEEESystemLoader
    # loader = IEEESystemLoader()
    # sim = loader.load_ieee68()
    # analyzer = SmallSignalStabilityAnalyzer(sim)
    # results = analyzer.assess_small_signal_stability()
    pass