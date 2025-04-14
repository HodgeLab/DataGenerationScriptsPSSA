"""
Module for performing small-signal stability analysis in ANDES.
"""

import logging
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_small_signal_analysis(system):
    """
    Perform small-signal stability analysis on an ANDES system.
    
    Args:
        system (andes.System): ANDES system object with a solved power flow
        
    Returns:
        bool: True if system is small-signal stable, False otherwise
        dict: Small-signal analysis results
    """
    try:
        # Create a deep copy to avoid modifying the original system
        system_copy = deepcopy(system)
        
        # Initialize and run power flow if not already run
        if not hasattr(system_copy, 'f') or not hasattr(system_copy.f, 'Bus'):
            logger.info("Running power flow before small-signal stability analysis")
            system_copy.PFlow.run()
            
            if not system_copy.PFlow.converged:
                logger.error("Power flow did not converge, cannot proceed with small-signal analysis")
                return False, {'error': 'Power flow did not converge'}
        
        # Run eigenvalue analysis
        logger.info("Running small-signal stability analysis")
        
        # Initialize small-signal module
        system_copy.SSSA.init()
        
        # Set up small-signal computation options
        system_copy.SSSA.config.eigs = True     # compute eigenvalues
        system_copy.SSSA.config.part_factor = 1 # compute participation factors
        
        # Run the small-signal analysis
        system_copy.SSSA.run()
        
        # Extract results
        eigenvalues = system_copy.SSSA.eigenvalues
        damping_ratios = system_copy.SSSA.damping
        frequencies = system_copy.SSSA.freq
        participation_factors = system_copy.SSSA.pf
        state_names = system_copy.SSSA.state_names if hasattr(system_copy.SSSA, 'state_names') else None
        
        # Organize eigenvalue results
        eigenvalue_results = []
        
        for i in range(len(eigenvalues)):
            eig_value = eigenvalues[i]
            damping = damping_ratios[i] * 100  # Convert to percentage
            freq = frequencies[i]
            
            # Get the top participating states if available
            top_states = {}
            if participation_factors is not None and state_names is not None:
                # Get participation factors for this eigenvalue
                pf_for_this_eig = participation_factors[:, i]
                
                # Sort by participation factor and get top 3
                top_indices = np.argsort(-np.abs(pf_for_this_eig))[:3]
                
                for idx in top_indices:
                    if idx < len(state_names):
                        state_name = state_names[idx]
                        pf_value = abs(pf_for_this_eig[idx])
                        top_states[state_name] = pf_value
            
            eigenvalue_results.append({
                'eigenvalue_idx': i,
                'real_part': eig_value.real,
                'imag_part': eig_value.imag,
                'frequency_hz': freq,
                'damping_ratio_percent': damping,
                'is_stable': eig_value.real < 0,
                'top_participating_states': top_states
            })
        
        # Convert to DataFrame for easier manipulation
        eigenvalue_df = pd.DataFrame(eigenvalue_results)
        
        # Focus on inter-area oscillation modes (0.25-1.0 Hz)
        inter_area_modes = eigenvalue_df[
            (eigenvalue_df['frequency_hz'] >= 0.25) & 
            (eigenvalue_df['frequency_hz'] <= 1.0)
        ]
        
        # Check if the system satisfies the 3% damping ratio criterion for inter-area modes
        min_damping_ratio = float('inf') if len(inter_area_modes) == 0 else inter_area_modes['damping_ratio_percent'].min()
        is_stable = min_damping_ratio >= 3.0
        
        # Identify poorly damped modes
        poorly_damped_modes = inter_area_modes[inter_area_modes['damping_ratio_percent'] < 3.0]
        
        # Create a summary of results
        results = {
            'is_stable': is_stable,
            'min_damping_ratio': min_damping_ratio if min_damping_ratio != float('inf') else None,
            'eigenvalues': eigenvalue_df.to_dict('records'),
            'inter_area_modes_count': len(inter_area_modes),
            'poorly_damped_modes_count': len(poorly_damped_modes),
            'poorly_damped_modes': poorly_damped_modes.to_dict('records') if len(poorly_damped_modes) > 0 else []
        }
        
        # Log summary of results
        if is_stable:
            logger.info(f"System is small-signal stable with minimum damping ratio of {min_damping_ratio:.2f}%")
        else:
            logger.warning(f"System is small-signal unstable with minimum damping ratio of {min_damping_ratio:.2f}%")
        
        return is_stable, results
    
    except Exception as e:
        logger.error(f"Failed to run small-signal stability analysis: {str(e)}")
        raise

def plot_eigenvalues(eigenvalue_results, min_damping_line=3.0, filename=None):
    """
    Plot the eigenvalues from small-signal stability analysis.
    
    Args:
        eigenvalue_results (dict): Results from small-signal stability analysis
        min_damping_line (float): Minimum required damping ratio in percentage
        filename (str, optional): If provided, save the plot to this file
        
    Returns:
        matplotlib.figure.Figure: Figure object with the eigenvalue plot
    """
    try:
        # Extract eigenvalues
        eigenvalues = np.array([(e['real_part'], e['imag_part']) for e in eigenvalue_results['eigenvalues']])
        damping_ratios = np.array([e['damping_ratio_percent'] for e in eigenvalue_results['eigenvalues']])
        frequencies = np.array([e['frequency_hz'] for e in eigenvalue_results['eigenvalues']])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all eigenvalues
        scatter = ax.scatter(eigenvalues[:, 0], eigenvalues[:, 1], 
                             c=damping_ratios, cmap='jet', s=30, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Damping Ratio (%)')
        
        # Plot constant damping lines
        r_max = max(abs(np.min(eigenvalues[:, 0])), abs(np.max(eigenvalues[:, 0])))
        r_max = max(r_max, 2)  # Ensure a minimum range
        y_max = max(abs(np.min(eigenvalues[:, 1])), abs(np.max(eigenvalues[:, 1])))
        y_max = max(y_max, 10)  # Ensure a minimum range
        
        # Add min damping ratio line
        zeta = min_damping_line / 100.0
        theta = np.arccos(zeta)
        x_line = np.linspace(-r_max, 0, 100)
        y_line_upper = np.tan(theta) * (-x_line)
        y_line_lower = -np.tan(theta) * (-x_line)
        
        ax.plot(x_line, y_line_upper, 'r--', linewidth=1.5, label=f'{min_damping_line}% Damping Ratio')
        ax.plot(x_line, y_line_lower, 'r--', linewidth=1.5)
        
        # Highlight inter-area modes
        inter_area_indices = [i for i, f in enumerate(frequencies) if 0.25 <= f <= 1.0]
        if inter_area_indices:
            ax.scatter(eigenvalues[inter_area_indices, 0], eigenvalues[inter_area_indices, 1], 
                       s=100, facecolors='none', edgecolors='r', linewidth=2, label='Inter-area Modes')
        
        # Add labels and title
        ax.set_xlabel('Real Part (1/s)')
        ax.set_ylabel('Imaginary Part (rad/s)')
        ax.set_title('Eigenvalue Plot with Damping Ratio Lines')
        
        # Add grid and legend
        ax.grid(True)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.legend()
        
        # Set equal aspect ratio and limits
        ax.set_xlim([-r_max, r_max/4])
        ax.set_ylim([-y_max, y_max])
        
        # Save if filename is provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Eigenvalue plot saved to {filename}")
        
        return fig
    
    except Exception as e:
        logger.error(f"Failed to plot eigenvalues: {str(e)}")
        raise

def identify_critical_modes(eigenvalue_results, damping_threshold=3.0):
    """
    Identify critical modes with low damping that may cause stability issues.
    
    Args:
        eigenvalue_results (dict): Results from small-signal stability analysis
        damping_threshold (float): Threshold for identifying critical modes (percentage)
        
    Returns:
        pd.DataFrame: DataFrame with critical modes information
    """
    try:
        # Extract eigenvalues as DataFrame
        eigenvalue_df = pd.DataFrame(eigenvalue_results['eigenvalues'])
        
        # Filter for oscillatory modes (non-zero frequency)
        oscillatory_modes = eigenvalue_df[eigenvalue_df['frequency_hz'] > 0]
        
        # Sort by damping ratio (ascending)
        sorted_modes = oscillatory_modes.sort_values('damping_ratio_percent')
        
        # Identify critical modes with damping below threshold
        critical_modes = sorted_modes[sorted_modes['damping_ratio_percent'] < damping_threshold]
        
        # Categorize modes by frequency range
        def categorize_mode(freq):
            if freq < 0.25:
                return 'Local'
            elif freq <= 1.0:
                return 'Inter-area'
            elif freq <= 2.0:
                return 'Control'
            else:
                return 'High-frequency'
        
        critical_modes['mode_type'] = critical_modes['frequency_hz'].apply(categorize_mode)
        
        # Log summary of critical modes
        if len(critical_modes) > 0:
            logger.warning(f"Identified {len(critical_modes)} critical modes with damping ratio below {damping_threshold}%")
            
            # Count by mode type
            mode_type_counts = critical_modes['mode_type'].value_counts()
            for mode_type, count in mode_type_counts.items():
                logger.warning(f"  - {count} {mode_type} modes")
        else:
            logger.info(f"No critical modes found with damping ratio below {damping_threshold}%")
        
        return critical_modes
    
    except Exception as e:
        logger.error(f"Failed to identify critical modes: {str(e)}")
        raise

def perform_small_signal_assessment(system, damping_threshold=3.0):
    """
    Perform comprehensive small-signal stability assessment.
    
    Args:
        system (andes.System): ANDES system object with a solved power flow
        damping_threshold (float): Threshold for identifying critical modes (percentage)
        
    Returns:
        dict: Dictionary with assessment results
    """
    try:
        # Run small-signal analysis
        logger.info("Performing small-signal stability assessment")
        is_stable, results = run_small_signal_analysis(system)
        
        # Identify critical modes
        critical_modes = identify_critical_modes(results, damping_threshold)
        
        # Prepare assessment results
        assessment_results = {
            'is_stable': is_stable,
            'min_damping_ratio': results['min_damping_ratio'],
            'critical_modes_count': len(critical_modes),
            'critical_modes': critical_modes.to_dict('records') if len(critical_modes) > 0 else [],
            'inter_area_modes_count': results['inter_area_modes_count'],
            'poorly_damped_modes_count': results['poorly_damped_modes_count'],
            'eigenvalue_stats': {
                'total_count': len(results['eigenvalues']),
                'unstable_count': sum(1 for e in results['eigenvalues'] if not e['is_stable']),
                'stable_count': sum(1 for e in results['eigenvalues'] if e['is_stable'])
            }
        }
        
        return assessment_results
    
    except Exception as e:
        logger.error(f"Failed to perform small-signal stability assessment: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    import load_ieee_system
    
    # Load IEEE 68-bus system
    system = load_ieee_system.load_ieee68()
    
    # Run power flow (required before small-signal analysis)
    system.PFlow.run()
    
    if system.PFlow.converged:
        # Perform small-signal assessment
        assessment = perform_small_signal_assessment(system, damping_threshold=3.0)
        
        # Print summary
        print("\nSmall-Signal Stability Assessment Summary:")
        print(f"Is stable: {assessment['is_stable']}")
        print(f"Minimum damping ratio: {assessment['min_damping_ratio']:.2f}%")
        print(f"Number of critical modes: {assessment['critical_modes_count']}")
        print(f"Number of inter-area modes: {assessment['inter_area_modes_count']}")
        print(f"Eigenvalue statistics: {assessment['eigenvalue_stats']}")
    else:
        print("Power flow did not converge, cannot perform small-signal analysis")