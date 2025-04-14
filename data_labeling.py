"""
Module for labeling power system stability data based on multiple criteria.
"""

import logging
import numpy as np
import pandas as pd
from copy import deepcopy
import os
import json
from datetime import datetime

# Import other modules
import load_ieee_system
import static_security
import small_signal_stability
import voltage_stability
import transient_stability
import fault_injection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StabilityLabeler:
    """
    Class for performing comprehensive stability assessment and data labeling.
    """
    
    def __init__(self, system=None, output_dir='./labeled_data'):
        """
        Initialize the StabilityLabeler.
        
        Args:
            system (andes.System, optional): ANDES system object
            output_dir (str): Directory to save labeled data
        """
        self.system = system
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize dataframe for storing labeled data
        self.data = pd.DataFrame()
        
        # Define stability criteria thresholds
        self.criteria = {
            'transient': {
                'tsi_threshold': 10.0  # TSI threshold in percentage
            },
            'small_signal': {
                'damping_threshold': 3.0  # Damping ratio threshold in percentage
            },
            'voltage': {
                'v_min': 0.8,  # Minimum voltage in p.u.
                'v_max': 1.1,  # Maximum voltage in p.u.
                'min_duration': 0.5  # Minimum duration for voltage violation in seconds
            },
            'static': {
                'overload_threshold': 100.0  # Line loading threshold in percentage
            }
        }
        
        logger.info("StabilityLabeler initialized")
    
    def load_system(self, system_name='ieee68', case_path=None):
        """
        Load a power system model.
        
        Args:
            system_name (str): Name of the system to load ('ieee68' or 'ieee300')
            case_path (str, optional): Path to a custom case file
            
        Returns:
            bool: True if system was loaded successfully
        """
        try:
            if system_name.lower() == 'ieee68':
                self.system = load_ieee_system.load_ieee68()
                logger.info("Loaded IEEE 68-bus system")
            elif system_name.lower() == 'ieee300':
                self.system = load_ieee_system.load_ieee300()
                logger.info("Loaded IEEE 300-bus system")
            elif case_path:
                self.system = load_ieee_system.load_custom_case(case_path)
                logger.info(f"Loaded custom case from {case_path}")
            else:
                logger.error("No valid system specified")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to load system: {str(e)}")
            return False
    
    def assess_all_stability(self, system, fault_event=None, t_end=10.0):
        """
        Perform comprehensive stability assessment on a system.
        
        Args:
            system (andes.System): ANDES system to assess
            fault_event (callable, optional): Function to apply a fault event
            t_end (float): End time for time-domain simulations
            
        Returns:
            dict: Dictionary with assessment results for all stability types
        """
        try:
            # If fault event provided, apply it to a copy of the system
            system_to_assess = system
            if fault_event is not None:
                logger.info("Applying fault event")
                system_to_assess = fault_event(deepcopy(system))
            
            # Run power flow for static security assessment
            logger.info("Running power flow for static security assessment")
            converged, _, solved_system = static_security.run_power_flow(system_to_assess)
            
            if not converged:
                logger.warning("Power flow did not converge")
                return {
                    'converged': False,
                    'static_secure': False,
                    'small_signal_stable': False,
                    'voltage_stable': False,
                    'transient_stable': False,
                    'overall_stable': False,
                    'reason': 'Power flow did not converge'
                }
            
            # 1. Static security assessment
            logger.info("Performing static security assessment")
            static_assessment = {}
            
            # Calculate line loadings
            line_loadings = static_security.calculate_line_loadings(solved_system)
            overloaded_lines = line_loadings[line_loadings['loading_percent'] > self.criteria['static']['overload_threshold']]
            
            # Calculate overload index
            overload_idx, _ = static_security.calculate_overload_index(solved_system)
            
            # Check voltage violations
            voltage_violations = static_security.check_voltage_violations(
                solved_system, 
                v_min=self.criteria['voltage']['v_min'], 
                v_max=self.criteria['voltage']['v_max']
            )
            
            static_assessment = {
                'static_secure': len(overloaded_lines) == 0,
                'overload_index': overload_idx,
                'overloaded_lines_count': len(overloaded_lines),
                'voltage_violations_count': len(voltage_violations)
            }
            
            # 2. Small-signal stability assessment
            logger.info("Performing small-signal stability assessment")
            ss_assessment = small_signal_stability.perform_small_signal_assessment(
                solved_system, 
                damping_threshold=self.criteria['small_signal']['damping_threshold']
            )
            
            # 3. Run time-domain simulation for both voltage and transient stability if needed
            logger.info("Running time-domain simulation for transient and voltage stability assessment")
            simulation_success, system_with_tds = transient_stability.run_transient_simulation(
                system_to_assess, t_end=t_end
            )
            
            if not simulation_success:
                logger.warning("Time-domain simulation failed")
                return {
                    'converged': True,
                    'static_secure': static_assessment['static_secure'],
                    'small_signal_stable': ss_assessment['is_stable'],
                    'voltage_stable': False,
                    'transient_stable': False,
                    'overall_stable': False,
                    'reason': 'Time-domain simulation failed',
                    'static_assessment': static_assessment,
                    'small_signal_assessment': ss_assessment
                }
            
            # 4. Voltage stability assessment
            logger.info("Performing voltage stability assessment")
            voltage_stable, voltage_assessment = voltage_stability.assess_voltage_stability(
                system_with_tds,
                v_min=self.criteria['voltage']['v_min'],
                v_max=self.criteria['voltage']['v_max'],
                min_duration=self.criteria['voltage']['min_duration']
            )
            
            # 5. Transient stability assessment
            logger.info("Performing transient stability assessment")
            transient_stable, transient_assessment = transient_stability.assess_transient_stability(
                system_with_tds,
                tsi_threshold=self.criteria['transient']['tsi_threshold']
            )
            
            # Combine all assessments
            overall_stable = (
                static_assessment['static_secure'] and
                ss_assessment['is_stable'] and
                voltage_stable and
                transient_stable
            )
            
            assessment_results = {
                'converged': True,
                'static_secure': static_assessment['static_secure'],
                'small_signal_stable': ss_assessment['is_stable'],
                'voltage_stable': voltage_stable,
                'transient_stable': transient_stable,
                'overall_stable': overall_stable,
                'static_assessment': static_assessment,
                'small_signal_assessment': ss_assessment,
                'voltage_assessment': voltage_assessment,
                'transient_assessment': transient_assessment
            }
            
            logger.info(f"Comprehensive stability assessment completed. System is {'stable' if overall_stable else 'unstable'}")
            return assessment_results
        
        except Exception as e:
            logger.error(f"Failed to perform comprehensive stability assessment: {str(e)}")
            raise
    
    def generate_labeled_dataset(self, n_scenarios=100, fault_types=None, save=True):
        """
        Generate a labeled dataset with multiple fault scenarios.
        
        Args:
            n_scenarios (int): Number of fault scenarios to generate
            fault_types (list, optional): List of fault types to include
            save (bool): Whether to save the dataset
            
        Returns:
            pd.DataFrame: DataFrame with labeled stability data
        """
        try:
            if self.system is None:
                logger.error("No system loaded. Call load_system() first.")
                return None
            
            # Generate fault scenarios
            logger.info(f"Generating {n_scenarios} fault scenarios")
            scenarios = fault_injection.generate_fault_scenarios(
                self.system, n_scenarios=n_scenarios, fault_types=fault_types
            )
            
            # Initialize data storage
            data_rows = []
            
            # Process each scenario
            for i, (faulted_system, fault_info) in enumerate(scenarios):
                logger.info(f"Processing scenario {i+1}/{n_scenarios}")
                
                # Assess stability
                assessment = self.assess_all_stability(faulted_system)
                
                # Create data row
                data_row = {
                    'scenario_id': i,
                    'fault_type': fault_info['type'],
                    'fault_time': fault_info['time'],
                    'clear_time': fault_info['clear_time'],
                    'converged': assessment['converged'],
                    'static_secure': assessment['static_secure'],
                    'small_signal_stable': assessment['small_signal_stable'],
                    'voltage_stable': assessment['voltage_stable'],
                    'transient_stable': assessment['transient_stable'],
                    'overall_stable': assessment['overall_stable']
                }
                
                # Add fault-specific details
                for key, value in fault_info.items():
                    if key not in data_row:
                        data_row[f'fault_{key}'] = value
                
                # Add detailed assessment metrics
                if assessment['converged']:
                    # Static security metrics
                    data_row['overload_index'] = assessment['static_assessment']['overload_index']
                    data_row['overloaded_lines_count'] = assessment['static_assessment']['overloaded_lines_count']
                    
                    # Small-signal stability metrics
                    data_row['min_damping_ratio'] = assessment['small_signal_assessment']['min_damping_ratio']
                    data_row['critical_modes_count'] = assessment['small_signal_assessment']['critical_modes_count']
                    
                    if assessment['transient_stable'] is not None:
                        # Transient stability metrics
                        data_row['tsi'] = assessment['transient_assessment']['tsi']
                        data_row['max_angular_separation'] = assessment['transient_assessment']['max_angular_separation']
                    
                    if assessment['voltage_stable'] is not None:
                        # Voltage stability metrics
                        data_row['voltage_violations_count'] = assessment['voltage_assessment']['violations_count']
                
                data_rows.append(data_row)
                
                # Log progress
                if (i + 1) % 10 == 0 or i + 1 == n_scenarios:
                    logger.info(f"Processed {i + 1}/{n_scenarios} scenarios")
            
            # Create DataFrame
            self.data = pd.DataFrame(data_rows)
            
            # Save dataset if requested
            if save:
                self._save_dataset()
            
            logger.info(f"Generated labeled dataset with {len(self.data)} scenarios")
            return self.data
        
        except Exception as e:
            logger.error(f"Failed to generate labeled dataset: {str(e)}")
            raise
    
    def _save_dataset(self, filename=None):
        """
        Save the labeled dataset to file.
        
        Args:
            filename (str, optional): Custom filename to use
            
        Returns:
            str: Path to the saved file
        """
        try:
            if self.data.empty:
                logger.warning("No data to save")
                return None
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"stability_data_{timestamp}.csv"
            
            # Ensure the filename has .csv extension
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            # Create full path
            filepath = os.path.join(self.output_dir, filename)
            
            # Save to CSV
            self.data.to_csv(filepath, index=False)
            logger.info(f"Saved labeled dataset to {filepath}")
            
            # Also save a metadata file with the criteria used
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'n_scenarios': len(self.data),
                'stability_criteria': self.criteria,
                'statistics': {
                    'overall_stable_percent': (self.data['overall_stable'].mean() * 100),
                    'static_secure_percent': (self.data['static_secure'].mean() * 100),
                    'small_signal_stable_percent': (self.data['small_signal_stable'].mean() * 100),
                    'voltage_stable_percent': (self.data['voltage_stable'].mean() * 100),
                    'transient_stable_percent': (self.data['transient_stable'].mean() * 100),
                }
            }
            
            metadata_filepath = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}_metadata.json")
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved metadata to {metadata_filepath}")
            
            return filepath
        
        except Exception as e:
            logger.error(f"Failed to save dataset: {str(e)}")
            return None
    
    def load_dataset(self, filepath):
        """
        Load a previously saved labeled dataset.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            bool: True if dataset was loaded successfully
        """
        try:
            self.data = pd.read_csv(filepath)
            logger.info(f"Loaded dataset with {len(self.data)} rows from {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            return False
    
    def get_stability_statistics(self):
        """
        Get statistics about the stability assessments in the dataset.
        
        Returns:
            dict: Dictionary with stability statistics
        """
        if self.data.empty:
            logger.warning("No data available for statistics")
            return {}
        
        try:
            stats = {
                'total_scenarios': len(self.data),
                'overall_stable_count': self.data['overall_stable'].sum(),
                'overall_stable_percent': (self.data['overall_stable'].mean() * 100),
                'static_secure_percent': (self.data['static_secure'].mean() * 100),
                'small_signal_stable_percent': (self.data['small_signal_stable'].mean() * 100),
                'voltage_stable_percent': (self.data['voltage_stable'].mean() * 100),
                'transient_stable_percent': (self.data['transient_stable'].mean() * 100),
            }
            
            # Add fault type breakdown
            fault_type_counts = self.data['fault_type'].value_counts()
            stats['fault_type_counts'] = fault_type_counts.to_dict()
            
            # Add stability by fault type
            stability_by_fault_type = {}
            for fault_type in fault_type_counts.index:
                fault_data = self.data[self.data['fault_type'] == fault_type]
                stability_by_fault_type[fault_type] = {
                    'count': len(fault_data),
                    'overall_stable_percent': (fault_data['overall_stable'].mean() * 100),
                    'static_secure_percent': (fault_data['static_secure'].mean() * 100),
                    'small_signal_stable_percent': (fault_data['small_signal_stable'].mean() * 100),
                    'voltage_stable_percent': (fault_data['voltage_stable'].mean() * 100),
                    'transient_stable_percent': (fault_data['transient_stable'].mean() * 100),
                }
            
            stats['stability_by_fault_type'] = stability_by_fault_type
            
            logger.info(f"Generated stability statistics")
            return stats
        
        except Exception as e:
            logger.error(f"Failed to generate statistics: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage
    labeler = StabilityLabeler(output_dir='./stability_data')
    
    # Load a system
    labeler.load_system('ieee68')
    
    # Generate a small labeled dataset
    data = labeler.generate_labeled_dataset(n_scenarios=5, save=True)
    
    # Print statistics
    stats = labeler.get_stability_statistics()
    print("\nStability Statistics:")
    for key, value in stats.items():
        if key != 'stability_by_fault_type' and key != 'fault_type_counts':
            print(f"  {key}: {value}")
    
    print("\nFault Type Counts:")
    for fault_type, count in stats.get('fault_type_counts', {}).items():
        print(f"  {fault_type}: {count}")
    
    print("\nStability by Fault Type:")
    for fault_type, fault_stats in stats.get('stability_by_fault_type', {}).items():
        print(f"  {fault_type}:")
        for stat_key, stat_value in fault_stats.items():
            print(f"    {stat_key}: {stat_value}")