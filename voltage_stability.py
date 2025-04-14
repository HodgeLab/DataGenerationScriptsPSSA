"""
Module for performing voltage stability assessment in ANDES.
"""

import logging
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_time_domain_simulation(system, t_end=10.0, dt=0.01):
    """
    Run time-domain simulation to assess voltage stability.
    
    Args:
        system (andes.System): ANDES system object with a solved power flow
        t_end (float): End time for simulation in seconds
        dt (float): Time step for simulation in seconds
        
    Returns:
        bool: True if simulation completed successfully, False otherwise
        dict: Simulation results
    """
    try:
        # Create a deep copy to avoid modifying the original system
        system_copy = deepcopy(system)
        
        # Initialize and run power flow if not already run
        if not hasattr(system_copy, 'f') or not hasattr(system_copy.f, 'Bus'):
            logger.info("Running power flow before time-domain simulation")
            system_copy.PFlow.run()
            
            if not system_copy.PFlow.converged:
                logger.error("Power flow did not converge, cannot proceed with time-domain simulation")
                return False, {'error': 'Power flow did not converge'}
        
        # Initialize time-domain simulation
        logger.info(f"Initializing time-domain simulation (t_end={t_end}s, dt={dt}s)")
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
            logger.info(f"Time-domain simulation completed successfully")
        else:
            logger.warning(f"Time-domain simulation failed to complete")
        
        # Return simulation results
        return success, system_copy
    
    except Exception as e:
        logger.error(f"Failed to run time-domain simulation: {str(e)}")
        raise

def assess_voltage_stability(system_with_tds, v_min=0.8, v_max=1.1, min_duration=0.5):
    """
    Assess voltage stability based on time-domain simulation results.
    A system is considered voltage unstable if any bus voltage deviates 
    from the range [v_min, v_max] for more than min_duration seconds.
    
    Args:
        system_with_tds (andes.System): ANDES system object with completed TDS simulation
        v_min (float): Minimum acceptable voltage in p.u.
        v_max (float): Maximum acceptable voltage in p.u.
        min_duration (float): Minimum duration for voltage violation in seconds
        
    Returns:
        bool: True if system is voltage stable, False otherwise
        dict: Voltage stability assessment results
    """
    try:
        # Check if time-domain simulation was run
        if not hasattr(system_with_tds, 'TDS') or not hasattr(system_with_tds.TDS, 'tv'):
            raise ValueError("Time-domain simulation results not available")
        
        # Get time vector and voltage data
        time_vector = system_with_tds.TDS.tv
        
        # Get bus voltage magnitudes
        n_buses = system_with_tds.Bus.n
        v_magnitudes = np.zeros((len(time_vector), n_buses))
        
        for t_idx, _ in enumerate(time_vector):
            for bus_idx in range(n_buses):
                v_magnitudes[t_idx, bus_idx] = abs(system_with_tds.TDS.y[t_idx][f'Bus_{bus_idx}_v'])
        
        # Calculate time step
        dt = time_vector[1] - time_vector[0] if len(time_vector) > 1 else 0.01
        
        # Analyze voltage violations for each bus
        violations = []
        
        for bus_idx in range(n_buses):
            v_bus = v_magnitudes[:, bus_idx]
            
            # Find periods where voltage is outside the acceptable range
            low_v_violation = v_bus < v_min
            high_v_violation = v_bus > v_max
            any_violation = np.logical_or(low_v_violation, high_v_violation)
            
            # Count consecutive violations
            if np.any(any_violation):
                consecutive_violations = 0
                max_consecutive = 0
                violation_start_time = None
                violation_periods = []
                
                for t_idx, is_violated in enumerate(any_violation):
                    if is_violated:
                        consecutive_violations += 1
                        
                        if consecutive_violations == 1:
                            violation_start_time = time_vector[t_idx]
                    else:
                        if consecutive_violations > 0:
                            violation_duration = consecutive_violations * dt
                            if violation_duration >= min_duration:
                                violation_end_time = time_vector[t_idx - 1]
                                violation_periods.append({
                                    'start_time': violation_start_time,
                                    'end_time': violation_end_time,
                                    'duration': violation_duration
                                })
                            
                            max_consecutive = max(max_consecutive, consecutive_violations)
                            consecutive_violations = 0
                
                # Check for violation at the end of simulation
                if consecutive_violations > 0:
                    violation_duration = consecutive_violations * dt
                    if violation_duration >= min_duration:
                        violation_end_time = time_vector[-1]
                        violation_periods.append({
                            'start_time': violation_start_time,
                            'end_time': violation_end_time,
                            'duration': violation_duration
                        })
                    
                    max_consecutive = max(max_consecutive, consecutive_violations)
                
                # If any violation period exists, add to the list
                if violation_periods:
                    bus_name = system_with_tds.Bus.name[bus_idx] if system_with_tds.Bus.name[bus_idx] else f"Bus_{bus_idx}"
                    
                    # Get min and max voltage during simulation
                    min_voltage = np.min(v_bus)
                    max_voltage = np.max(v_bus)
                    
                    violations.append({
                        'bus_idx': bus_idx,
                        'bus_name': bus_name,
                        'min_voltage': min_voltage,
                        'max_voltage': max_voltage,
                        'max_violation_duration': max(period['duration'] for period in violation_periods),
                        'total_violation_duration': sum(period['duration'] for period in violation_periods),
                        'violation_periods': violation_periods
                    })
        
        # Create a DataFrame for the violations
        violations_df = pd.DataFrame(violations) if violations else pd.DataFrame()
        
        # Determine if system is voltage stable
        is_stable = len(violations) == 0
        
        # Create assessment results
        assessment_results = {
            'is_stable': is_stable,
            'violations_count': len(violations),
            'violations': violations_df.to_dict('records') if len(violations_df) > 0 else [],
            'v_min': v_min,
            'v_max': v_max,
            'min_duration': min_duration
        }
        
        if is_stable:
            logger.info("System is voltage stable")
        else:
            logger.warning(f"System is voltage unstable with {len(violations)} bus violations")
        
        return is_stable, assessment_results
    
    except Exception as e:
        logger.error(f"Failed to assess voltage stability: {str(e)}")
        raise

def plot_voltage_profiles(system_with_tds, bus_indices=None, filename=None):
    """
    Plot voltage profiles for selected buses from time-domain simulation.
    
    Args:
        system_with_tds (andes.System): ANDES system object with completed TDS simulation
        bus_indices (list, optional): List of bus indices to plot. If None, plot all buses
        filename (str, optional): If provided, save the plot to this file
        
    Returns:
        matplotlib.figure.Figure: Figure object with the voltage profiles plot
    """
    try:
        # Check if time-domain simulation was run
        if not hasattr(system_with_tds, 'TDS') or not hasattr(system_with_tds.TDS, 'tv'):
            raise ValueError("Time-domain simulation results not available")
        
        # Get time vector
        time_vector = system_with_tds.TDS.tv
        
        # Determine which buses to plot
        if bus_indices is None:
            # If no specific buses, plot all
            bus_indices = list(range(system_with_tds.Bus.n))
        elif len(bus_indices) > 10:
            # If too many buses are specified, limit to 10
            logger.warning(f"Too many buses selected. Limiting to the first 10.")
            bus_indices = bus_indices[:10]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot voltage for each selected bus
        for bus_idx in bus_indices:
            if bus_idx >= system_with_tds.Bus.n:
                logger.warning(f"Bus index {bus_idx} is out of range")
                continue
            
            # Get bus name
            bus_name = system_with_tds.Bus.name[bus_idx] if system_with_tds.Bus.name[bus_idx] else f"Bus_{bus_idx}"
            
            # Extract voltage magnitude
            v_magnitude = np.array([abs(system_with_tds.TDS.y[t_idx][f'Bus_{bus_idx}_v']) 
                                  for t_idx in range(len(time_vector))])
            
            # Plot voltage profile
            ax.plot(time_vector, v_magnitude, label=f"Bus {bus_name}")
        
        # Add horizontal lines for typical voltage limits
        ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Min Voltage (0.8 pu)')
        ax.axhline(y=1.1, color='r', linestyle='--', alpha=0.5, label='Max Voltage (1.1 pu)')
        
        # Add labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage Magnitude (p.u.)')
        ax.set_title('Bus Voltage Profiles during Simulation')
        
        # Add grid and legend
        ax.grid(True)
        ax.legend(loc='best')
        
        # Save if filename is provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Voltage profiles plot saved to {filename}")
        
        return fig
    
    except Exception as e:
        logger.error(f"Failed to plot voltage profiles: {str(e)}")
        raise

def perform_voltage_stability_assessment(system, fault_event=None, t_end=10.0, v_min=0.8, v_max=1.1, min_duration=0.5):
    """
    Perform comprehensive voltage stability assessment including time-domain simulation with a fault event.
    
    Args:
        system (andes.System): ANDES system object
        fault_event (callable, optional): Function to add a fault event to the system
        t_end (float): End time for simulation in seconds
        v_min (float): Minimum acceptable voltage in p.u.
        v_max (float): Maximum acceptable voltage in p.u.
        min_duration (float): Minimum duration for voltage violation in seconds
        
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
        logger.info("Running power flow before voltage stability assessment")
        system_copy.PFlow.run()
        
        if not system_copy.PFlow.converged:
            logger.error("Power flow did not converge, cannot proceed with voltage stability assessment")
            return {
                'is_stable': False,
                'reason': 'Power flow did not converge',
                'simulation_success': False
            }
        
        # Run time-domain simulation
        logger.info("Running time-domain simulation for voltage stability assessment")
        simulation_success, system_with_tds = run_time_domain_simulation(system_copy, t_end=t_end)
        
        if not simulation_success:
            logger.warning("Time-domain simulation failed, cannot assess voltage stability")
            return {
                'is_stable': False,
                'reason': 'Time-domain simulation failed',
                'simulation_success': False
            }
        
        # Assess voltage stability
        logger.info("Assessing voltage stability from simulation results")
        is_stable, assessment_results = assess_voltage_stability(
            system_with_tds, v_min=v_min, v_max=v_max, min_duration=min_duration
        )
        
        # Add simulation success flag
        assessment_results['simulation_success'] = True
        
        return assessment_results
    
    except Exception as e:
        logger.error(f"Failed to perform voltage stability assessment: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    import load_ieee_system
    from fault_injection import apply_line_fault
    
    # Load IEEE 68-bus system
    system = load_ieee_system.load_ieee68()
    
    # Define a simple fault event (will be imported from fault_injection.py in real use)
    def example_fault(sys):
        # Create a copy of the system
        sys_copy = deepcopy(sys)
        # Apply a fault on bus 1 at t=1.0s, clear after 0.1s
        sys_copy.TDS.add_event('bus_fault', 
                             {'bus': 1, 'tf': 1.0, 'tc': 1.1, 'r': 0.0, 'x': 0.01})
        return sys_copy
    
    # Perform voltage stability assessment
    assessment = perform_voltage_stability_assessment(
        system, fault_event=example_fault, t_end=5.0, v_min=0.8, v_max=1.1, min_duration=0.5
    )
    
    # Print summary
    print("\nVoltage Stability Assessment Summary:")
    print(f"Simulation completed successfully: {assessment.get('simulation_success', False)}")
    print(f"System is voltage stable: {assessment.get('is_stable', False)}")
    print(f"Number of bus violations: {assessment.get('violations_count', 0)}")
    
    if not assessment.get('is_stable', False) and 'violations' in assessment:
        print("\nViolations:")
        for violation in assessment['violations']:
            print(f"  Bus {violation['bus_name']}:")
            print(f"    Min voltage: {violation['min_voltage']:.4f} pu")
            print(f"    Max voltage: {violation['max_voltage']:.4f} pu")
            print(f"    Max violation duration: {violation['max_violation_duration']:.2f} s")