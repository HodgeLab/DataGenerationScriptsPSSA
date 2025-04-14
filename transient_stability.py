"""
Module for performing transient stability analysis in ANDES.
"""

import logging
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_transient_simulation(system, t_end=10.0, dt=0.01):
    """
    Run time-domain simulation for transient stability analysis.
    
    Args:
        system (andes.System): ANDES system object with a solved power flow
        t_end (float): End time for simulation in seconds
        dt (float): Time step for simulation in seconds
        
    Returns:
        bool: True if simulation completed successfully, False otherwise
        andes.System: System object with simulation results
    """
    try:
        # Create a deep copy to avoid modifying the original system
        system_copy = deepcopy(system)
        
        # Initialize and run power flow if not already run
        if not hasattr(system_copy, 'f') or not hasattr(system_copy.f, 'Bus'):
            logger.info("Running power flow before transient simulation")
            system_copy.PFlow.run()
            
            if not system_copy.PFlow.converged:
                logger.error("Power flow did not converge, cannot proceed with transient simulation")
                return False, None
        
        # Initialize time-domain simulation
        logger.info(f"Initializing transient simulation (t_end={t_end}s, dt={dt}s)")
        system_copy.TDS.config.tf = t_end
        system_copy.TDS.config.h = dt
        system_copy.TDS.config.fixt = False  # Variable time step
        system_copy.TDS.config.tstep = 1     # Output sampling interval (multiply by dt)
        
        # Run time-domain simulation
        system_copy.TDS.init()
        system_copy.TDS.run()
        
        # Check if simulation completed successfully
        success = system_copy.TDS.converged
        
        if success:
            logger.info(f"Transient simulation completed successfully")
        else:
            logger.warning(f"Transient simulation failed to complete")
        
        # Return simulation results
        return success, system_copy
    
    except Exception as e:
        logger.error(f"Failed to run transient simulation: {str(e)}")
        raise

def get_generator_angles(system_with_tds):
    """
    Extract generator rotor angles from time-domain simulation results.
    
    Args:
        system_with_tds (andes.System): ANDES system object with completed TDS simulation
        
    Returns:
        pd.DataFrame: DataFrame with generator rotor angles over time
    """
    try:
        # Check if time-domain simulation was run
        if not hasattr(system_with_tds, 'TDS') or not hasattr(system_with_tds.TDS, 'tv'):
            raise ValueError("Time-domain simulation results not available")
        
        # Get time vector
        time_vector = system_with_tds.TDS.tv
        
        # Get generator rotor angles
        angles_data = defaultdict(list)
        angles_data['time'] = time_vector
        
        # Extract generator angles for all generators (GENROU model)
        if hasattr(system_with_tds, 'GENROU'):
            for gen_idx in range(system_with_tds.GENROU.n):
                gen_name = system_with_tds.GENROU.name[gen_idx] if system_with_tds.GENROU.name[gen_idx] else f"GENROU_{gen_idx}"
                
                # Extract angle data
                angles = []
                for t_idx in range(len(time_vector)):
                    # Convert to degrees for analysis
                    angle_rad = system_with_tds.TDS.y[t_idx][f'GENROU_{gen_idx}_delta']
                    angle_deg = np.degrees(angle_rad) % 360
                    angles.append(angle_deg)
                
                angles_data[gen_name] = angles
        
        # Create a DataFrame from the collected data
        angles_df = pd.DataFrame(angles_data)
        
        return angles_df
    
    except Exception as e:
        logger.error(f"Failed to extract generator angles: {str(e)}")
        raise

def calculate_transient_stability_index(angles_df):
    """
    Calculate the Transient Stability Index (TSI) based on maximum angular separation.
    TSI = (360 - δ_max) / (360 + δ_max) * 100%
    
    Args:
        angles_df (pd.DataFrame): DataFrame with generator rotor angles
        
    Returns:
        float: Transient Stability Index in percentage
        float: Maximum angular separation in degrees
    """
    try:
        # Get angle data (skip the time column)
        angle_columns = [col for col in angles_df.columns if col != 'time']
        
        # Calculate maximum angular separation for each time step
        max_separation = []
        
        for _, row in angles_df.iterrows():
            angles = [row[col] for col in angle_columns]
            if len(angles) >= 2:  # Need at least 2 generators
                # Calculate maximum separation between any two angles
                max_diff = 0
                for i in range(len(angles)):
                    for j in range(i+1, len(angles)):
                        # Calculate the shortest angle between two generators (considering 360° periodicity)
                        diff = min(abs(angles[i] - angles[j]), 360 - abs(angles[i] - angles[j]))
                        max_diff = max(max_diff, diff)
                
                max_separation.append(max_diff)
        
        # Get the maximum separation across all time steps
        delta_max = max(max_separation) if max_separation else 0
        
        # Calculate TSI
        tsi = (360 - delta_max) / (360 + delta_max) * 100
        
        logger.info(f"Maximum angular separation: {delta_max:.2f} degrees")
        logger.info(f"Transient Stability Index (TSI): {tsi:.2f}%")
        
        return tsi, delta_max
    
    except Exception as e:
        logger.error(f"Failed to calculate transient stability index: {str(e)}")
        raise

def assess_transient_stability(system_with_tds, tsi_threshold=10.0):
    """
    Assess transient stability based on the TSI threshold.
    The system is considered transiently stable if TSI >= tsi_threshold.
    
    Args:
        system_with_tds (andes.System): ANDES system object with completed TDS simulation
        tsi_threshold (float): Threshold for TSI in percentage (default: 10%)
        
    Returns:
        bool: True if system is transiently stable, False otherwise
        dict: Transient stability assessment results
    """
    try:
        # Extract generator angles
        angles_df = get_generator_angles(system_with_tds)
        
        # Calculate TSI
        tsi, delta_max = calculate_transient_stability_index(angles_df)
        
        # Assess stability
        is_stable = tsi >= tsi_threshold
        
        # Create assessment results
        assessment_results = {
            'is_stable': is_stable,
            'tsi': tsi,
            'tsi_threshold': tsi_threshold,
            'max_angular_separation': delta_max,
            'generator_count': len(angles_df.columns) - 1  # Subtract the time column
        }
        
        if is_stable:
            logger.info(f"System is transiently stable (TSI: {tsi:.2f}% >= threshold: {tsi_threshold}%)")
        else:
            logger.warning(f"System is transiently unstable (TSI: {tsi:.2f}% < threshold: {tsi_threshold}%)")
        
        return is_stable, assessment_results
    
    except Exception as e:
        logger.error(f"Failed to assess transient stability: {str(e)}")
        raise

def plot_generator_angles(angles_df, filename=None):
    """
    Plot generator rotor angles from time-domain simulation.
    
    Args:
        angles_df (pd.DataFrame): DataFrame with generator rotor angles
        filename (str, optional): If provided, save the plot to this file
        
    Returns:
        matplotlib.figure.Figure: Figure object with the generator angles plot
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get angle columns (skip the time column)
        angle_columns = [col for col in angles_df.columns if col != 'time']
        
        # Plot each generator's angle
        for col in angle_columns:
            ax.plot(angles_df['time'], angles_df[col], label=col)
        
        # Add labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rotor Angle (degrees)')
        ax.set_title('Generator Rotor Angles during Transient Simulation')
        
        # Add grid and legend
        ax.grid(True)
        ax.legend(loc='best', fontsize=8)
        
        # Save if filename is provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Generator angles plot saved to {filename}")
        
        return fig
    
    except Exception as e:
        logger.error(f"Failed to plot generator angles: {str(e)}")
        raise

def perform_transient_stability_assessment(system, fault_event=None, t_end=10.0, tsi_threshold=10.0):
    """
    Perform comprehensive transient stability assessment including time-domain simulation with a fault event.
    
    Args:
        system (andes.System): ANDES system object
        fault_event (callable, optional): Function to add a fault event to the system
        t_end (float): End time for simulation in seconds
        tsi_threshold (float): Threshold for TSI in percentage
        
    Returns:
        dict: Dictionary with assessment results
    """
    try:
        # Create a copy of the system
        system_copy = deepcopy(system)
        
        # Apply fault event if provided
        if fault_event is not None:
            logger.info("Applying fault event to the system")
            system_copy = fault_event(system_copy)
        
        # Run power flow to initialize the system
        logger.info("Running power flow before transient stability assessment")
        system_copy.PFlow.run()
        
        if not system_copy.PFlow.converged:
            logger.error("Power flow did not converge, cannot proceed with transient stability assessment")
            return {
                'is_stable': False,
                'reason': 'Power flow did not converge',
                'simulation_success': False
            }
        
        # Run transient simulation
        logger.info("Running transient simulation for stability assessment")
        simulation_success, system_with_tds = run_transient_simulation(system_copy, t_end=t_end)
        
        if not simulation_success:
            logger.warning("Transient simulation failed, cannot assess stability")
            return {
                'is_stable': False,
                'reason': 'Transient simulation failed',
                'simulation_success': False
            }
        
        # Assess transient stability
        logger.info("Assessing transient stability from simulation results")
        is_stable, assessment_results = assess_transient_stability(
            system_with_tds, tsi_threshold=tsi_threshold
        )
        
        # Add simulation success flag
        assessment_results['simulation_success'] = True
        
        return assessment_results
    
    except Exception as e:
        logger.error(f"Failed to perform transient stability assessment: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    import load_ieee_system
    
    # Load IEEE 68-bus system
    system = load_ieee_system.load_ieee68()
    
    # Define a simple fault event
    def example_fault(sys):
        # Create a copy of the system
        sys_copy = deepcopy(sys)
        # Apply a three-phase fault on bus 1 at t=1.0s, clear after 0.1s
        sys_copy.TDS.add_event('bus_fault', 
                             {'bus': 1, 'tf': 1.0, 'tc': 1.1, 'r': 0.0, 'x': 0.01})
        return sys_copy
    
    # Perform transient stability assessment
    assessment = perform_transient_stability_assessment(
        system, fault_event=example_fault, t_end=5.0, tsi_threshold=10.0
    )
    
    # Print summary
    print("\nTransient Stability Assessment Summary:")
    print(f"Simulation completed successfully: {assessment.get('simulation_success', False)}")
    
    if assessment.get('simulation_success', False):
        print(f"System is transiently stable: {assessment['is_stable']}")
        print(f"Transient Stability Index (TSI): {assessment['tsi']:.2f}%")
        print(f"Maximum angular separation: {assessment['max_angular_separation']:.2f} degrees")
        print(f"TSI threshold: {assessment['tsi_threshold']}%")
        print(f"Number of generators: {assessment['generator_count']}")
    else:
        print(f"Reason for assessment failure: {assessment.get('reason', 'Unknown')}")