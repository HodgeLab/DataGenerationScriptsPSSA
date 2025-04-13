"""
Module for labeling power system stability data based on stability assessment criteria.
"""

import numpy as np
import pandas as pd
from enum import Enum

# Import modules for stability assessment
from static_security import StaticSecurityAssessor
from small_signal_stability import SmallSignalStabilityAnalyzer
from voltage_stability import VoltageStabilityAnalyzer
from transient_stability import TransientStabilityAnalyzer


class StabilityLabels(Enum):
    """Enumeration of stability labels."""
    STABLE = "stable"
    UNSTABLE = "unstable"
    MARGINALLY_STABLE = "marginally_stable"
    UNKNOWN = "unknown"


class SecurityType(Enum):
    """Enumeration of security assessment types."""
    STATIC = "static_security"
    SMALL_SIGNAL = "small_signal_stability"
    VOLTAGE = "voltage_stability"
    TRANSIENT = "transient_stability"
    OVERALL = "overall_security"


class DataLabeler:
    """Class to label power system data based on various stability criteria."""
    
    def __init__(self, simulation):
        """
        Initialize the data labeler.
        
        Args:
            simulation: A Dynawo simulation object
        """
        self.simulation = simulation
        self.network = simulation.get_network()
        
        # Initialize assessors for each stability type
        self.static_assessor = StaticSecurityAssessor(simulation)
        self.small_signal_analyzer = SmallSignalStabilityAnalyzer(simulation)
        self.voltage_analyzer = VoltageStabilityAnalyzer(simulation)
        self.transient_analyzer = TransientStabilityAnalyzer(simulation)
        
        # Store assessment results
        self.results = {}
        
        # Define default criteria
        self.criteria = {
            'static_security': {
                'overload_threshold': 90  # Line loading percentage threshold
            },
            'small_signal_stability': {
                'min_damping': 0.03,  # Minimum damping ratio
                'freq_min': 0.25,     # Minimum frequency of interest (Hz)
                'freq_max': 1.0       # Maximum frequency of interest (Hz)
            },
            'voltage_stability': {
                'v_min': 0.8,            # Minimum acceptable voltage in p.u.
                'v_max': 1.1,            # Maximum acceptable voltage in p.u.
                'violation_duration': 0.5 # Maximum duration for voltage violations (seconds)
            },
            'transient_stability': {
                'tsi_threshold': 10.0   # TSI threshold to consider stable (%)
            }
        }
    
    def set_criteria(self, criteria_type, **kwargs):
        """
        Set custom criteria for stability assessment.
        
        Args:
            criteria_type: Type of stability criteria to set
            **kwargs: Criteria parameters
        """
        if criteria_type in self.criteria:
            for key, value in kwargs.items():
                if key in self.criteria[criteria_type]:
                    self.criteria[criteria_type][key] = value
                    print(f"Set {criteria_type}.{key} = {value}")
                else:
                    print(f"Warning: Unknown criterion '{key}' for {criteria_type}")
        else:
            print(f"Warning: Unknown criteria type '{criteria_type}'")
    
    def assess_static_security(self):
        """
        Perform static security assessment.
        
        Returns:
            Tuple of (label, details)
        """
        print("Assessing static security...")
        
        criteria = self.criteria['static_security']
        
        # Perform assessment
        assessment = self.static_assessor.assess_static_security(
            overload_threshold=criteria['overload_threshold']
        )
        
        # Determine label
        if not assessment['converged']:
            label = StabilityLabels.UNSTABLE.value
            details = "Power flow did not converge"
        elif assessment['secure']:
            label = StabilityLabels.STABLE.value
            details = "No violations found"
        else:
            # Calculate overload index
            if 'overload_index' in assessment:
                overload_index = assessment['overload_index']
                
                # Formula from the criteria:
                # f_x = sum(w_i * (S_mean,i / S_max,i)^p)
                # We consider the system marginally stable if 0.9 < overload_index < 1.0
                if overload_index is not None:
                    if overload_index < 0.8:
                        label = StabilityLabels.STABLE.value
                        details = f"Overload index is low: {overload_index:.4f}"
                    elif overload_index < 0.9:
                        label = StabilityLabels.MARGINALLY_STABLE.value
                        details = f"Overload index is moderate: {overload_index:.4f}"
                    else:
                        label = StabilityLabels.UNSTABLE.value
                        details = f"Overload index is high: {overload_index:.4f}"
                else:
                    label = StabilityLabels.UNSTABLE.value
                    details = "Critical line loadings or voltage violations"
            else:
                label = StabilityLabels.UNSTABLE.value
                details = "Critical line loadings or voltage violations"
        
        # Store results
        self.results[SecurityType.STATIC.value] = {
            'label': label,
            'details': details,
            'assessment': assessment
        }
        
        print(f"Static security assessment result: {label} - {details}")
        return label, details
    
    def assess_small_signal_stability(self):
        """
        Perform small-signal stability assessment.
        
        Returns:
            Tuple of (label, details)
        """
        print("Assessing small-signal stability...")
        
        criteria = self.criteria['small_signal_stability']
        
        # Perform assessment
        assessment = self.small_signal_analyzer.assess_small_signal_stability(
            min_damping=criteria['min_damping'],
            freq_min=criteria['freq_min'],
            freq_max=criteria['freq_max']
        )
        
        # Determine label
        if not assessment['success']:
            label = StabilityLabels.UNKNOWN.value
            details = "Linearization failed"
        elif assessment['stable']:
            label = StabilityLabels.STABLE.value
            details = "All modes have sufficient damping"
        else:
            # Check severity of damping issues
            if 'poorly_damped_modes' in assessment and assessment['poorly_damped_modes'] is not None:
                poorly_damped = assessment['poorly_damped_modes']
                
                if len(poorly_damped) == 0:
                    label = StabilityLabels.STABLE.value
                    details = "All modes have sufficient damping"
                else:
                    # Get minimum damping ratio
                    min_damping = poorly_damped['damping_ratio'].min()
                    
                    if min_damping >= 0.01:  # Still has some damping
                        label = StabilityLabels.MARGINALLY_STABLE.value
                        details = f"Minimum damping ratio: {min_damping:.4f} (threshold: {criteria['min_damping']})"
                    else:
                        label = StabilityLabels.UNSTABLE.value
                        details = f"Very poor damping: {min_damping:.4f} (threshold: {criteria['min_damping']})"
            else:
                label = StabilityLabels.UNSTABLE.value
                details = "Poorly damped oscillatory modes"
        
        # Store results
        self.results[SecurityType.SMALL_SIGNAL.value] = {
            'label': label,
            'details': details,
            'assessment': assessment
        }
        
        print(f"Small-signal stability assessment result: {label} - {details}")
        return label, details
    
    def assess_voltage_stability(self):
        """
        Perform voltage stability assessment.
        
        Returns:
            Tuple of (label, details)
        """
        print("Assessing voltage stability...")
        
        criteria = self.criteria['voltage_stability']
        
        # Perform assessment
        assessment = self.voltage_analyzer.assess_voltage_stability(
            v_min=criteria['v_min'],
            v_max=criteria['v_max'],
            violation_duration=criteria['violation_duration']
        )
        
        # Determine label
        if not assessment['success']:
            label = StabilityLabels.UNKNOWN.value
            details = "Simulation failed"
        elif assessment['secure']:
            label = StabilityLabels.STABLE.value
            details = "No sustained voltage violations"
        else:
            # Check severity of voltage violations
            num_violations = len(assessment['violations'])
            max_duration = 0
            extreme_voltage = 0
            
            for bus_id, violation in assessment['violations'].items():
                max_duration = max(max_duration, violation['duration'])
                
                if violation['violation_type'] == 'Low':
                    extreme_voltage = min(extreme_voltage, violation['min_voltage'])
                else:  # High violation
                    extreme_voltage = max(extreme_voltage, violation['max_voltage'])
            
            # Determine stability based on criteria specified in the requirements
            # "A system is considered insecure if any bus voltage deviates from the range of 0.8 pu to 1.1 pu for more than 0.5 seconds"
            if max_duration > criteria['violation_duration'] + 0.2:  # Significantly longer violation
                label = StabilityLabels.UNSTABLE.value
                details = f"{num_violations} buses with voltage violations for up to {max_duration:.2f}s"
            elif max_duration > criteria['violation_duration']:
                label = StabilityLabels.MARGINALLY_STABLE.value
                details = f"{num_violations} buses with voltage violations for up to {max_duration:.2f}s"
            else:
                label = StabilityLabels.STABLE.value
                details = "Short-duration voltage deviations only"
        
        # Store results
        self.results[SecurityType.VOLTAGE.value] = {
            'label': label,
            'details': details,
            'assessment': assessment
        }
        
        print(f"Voltage stability assessment result: {label} - {details}")
        return label, details
    
    def assess_transient_stability(self):
        """
        Perform transient stability assessment.
        
        Returns:
            Tuple of (label, details)
        """
        print("Assessing transient stability...")
        
        criteria = self.criteria['transient_stability']
        
        # Perform assessment
        assessment = self.transient_analyzer.assess_transient_stability(
            tsi_threshold=criteria['tsi_threshold']
        )
        
        # Determine label based on TSI value
        # "The system is considered transiently insecure if the TSI is less than 10%"
        if assessment.get('tsi') is None:
            label = StabilityLabels.UNKNOWN.value
            details = "Could not calculate TSI"
        else:
            tsi = assessment['tsi']
            
            if tsi >= criteria['tsi_threshold'] + 10:  # Well above threshold
                label = StabilityLabels.STABLE.value
                details = f"TSI = {tsi:.2f}% (threshold: {criteria['tsi_threshold']}%)"
            elif tsi >= criteria['tsi_threshold']:  # Just above threshold
                label = StabilityLabels.MARGINALLY_STABLE.value
                details = f"TSI = {tsi:.2f}% (threshold: {criteria['tsi_threshold']}%)"
            else:
                label = StabilityLabels.UNSTABLE.value
                details = f"TSI = {tsi:.2f}% (threshold: {criteria['tsi_threshold']}%)"
        
        # Store results
        self.results[SecurityType.TRANSIENT.value] = {
            'label': label,
            'details': details,
            'assessment': assessment
        }
        
        print(f"Transient stability assessment result: {label} - {details}")
        return label, details
    
    def assess_overall_stability(self):
        """
        Perform a comprehensive stability assessment.
        
        Returns:
            Tuple of (label, details)
        """
        print("Performing comprehensive stability assessment...")
        
        # Perform all individual assessments if not already done
        if SecurityType.STATIC.value not in self.results:
            self.assess_static_security()
        
        if SecurityType.SMALL_SIGNAL.value not in self.results:
            self.assess_small_signal_stability()
        
        if SecurityType.VOLTAGE.value not in self.results:
            self.assess_voltage_stability()
        
        if SecurityType.TRANSIENT.value not in self.results:
            self.assess_transient_stability()
        
        # Count results by label
        label_counts = {
            StabilityLabels.STABLE.value: 0,
            StabilityLabels.MARGINALLY_STABLE.value: 0,
            StabilityLabels.UNSTABLE.value: 0,
            StabilityLabels.UNKNOWN.value: 0
        }
        
        for security_type, result in self.results.items():
            if security_type != SecurityType.OVERALL.value:
                label_counts[result['label']] += 1
        
        # Determine overall label
        if label_counts[StabilityLabels.UNSTABLE.value] > 0:
            # Any unstable criterion makes the system unstable
            label = StabilityLabels.UNSTABLE.value
            details = "System is unstable in one or more criteria"
        elif label_counts[StabilityLabels.MARGINALLY_STABLE.value] > 0:
            # Any marginally stable criterion with no unstable ones
            label = StabilityLabels.MARGINALLY_STABLE.value
            details = "System is marginally stable in one or more criteria"
        elif label_counts[StabilityLabels.STABLE.value] + label_counts[StabilityLabels.UNKNOWN.value] == 4:
            # All criteria are either stable or unknown
            if label_counts[StabilityLabels.UNKNOWN.value] > 0:
                label = StabilityLabels.MARGINALLY_STABLE.value
                details = "System appears stable but some assessments are incomplete"
            else:
                label = StabilityLabels.STABLE.value
                details = "System is stable in all criteria"
        else:
            # Shouldn't reach here, but just in case
            label = StabilityLabels.UNKNOWN.value
            details = "Could not determine overall stability"
        
        # Store results
        self.results[SecurityType.OVERALL.value] = {
            'label': label,
            'details': details,
            'label_counts': label_counts
        }
        
        print(f"Overall stability assessment result: {label} - {details}")
        return label, details
    
    def get_stability_dataframe(self):
        """
        Get a DataFrame containing stability assessment results.
        
        Returns:
            DataFrame with stability assessment results
        """
        # Ensure all assessments have been performed
        if SecurityType.OVERALL.value not in self.results:
            self.assess_overall_stability()
        
        # Prepare data
        data = []
        
        for security_type, result in self.results.items():
            data.append({
                'security_type': security_type,
                'label': result['label'],
                'details': result['details']
            })
        
        return pd.DataFrame(data)
    
    def export_results(self, filename=None):
        """
        Export stability assessment results to a file.
        
        Args:
            filename: Name of the file to export to (if None, return a dictionary)
            
        Returns:
            Dictionary with results if filename is None, otherwise None
        """
        # Ensure all assessments have been performed
        if SecurityType.OVERALL.value not in self.results:
            self.assess_overall_stability()
        
        # Prepare export data
        export_data = {
            'overall': {
                'label': self.results[SecurityType.OVERALL.value]['label'],
                'details': self.results[SecurityType.OVERALL.value]['details']
            },
            'criteria': self.criteria,
            'assessments': {}
        }
        
        for security_type, result in self.results.items():
            if security_type != SecurityType.OVERALL.value:
                export_data['assessments'][security_type] = {
                    'label': result['label'],
                    'details': result['details']
                }
        
        # Export to file if specified
        if filename:
            if filename.endswith('.json'):
                import json
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
            elif filename.endswith('.csv'):
                df = self.get_stability_dataframe()
                df.to_csv(filename, index=False)
            else:
                print(f"Unsupported file format: {filename}")
                return export_data
            
            print(f"Results exported to {filename}")
            return None
        else:
            return export_data


# Example usage
if __name__ == "__main__":
    # This requires a simulation object to be created first
    # from load_ieee_systems import IEEESystemLoader
    # loader = IEEESystemLoader()
    # sim = loader.load_ieee68()
    # labeler = DataLabeler(sim)
    # labeler.assess_overall_stability()
    # df = labeler.get_stability_dataframe()
    # print(df)
    pass