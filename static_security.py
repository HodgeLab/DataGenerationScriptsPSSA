"""
Module for performing static security assessment through power flows in ANDES.
"""

import logging
import numpy as np
import pandas as pd
from copy import deepcopy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_power_flow(system):
    """
    Perform power flow analysis on an ANDES system.
    
    Args:
        system (andes.System): ANDES system object
        
    Returns:
        bool: True if power flow converged, False otherwise
        dict: Power flow results and statistics
    """
    try:
        # Create a deep copy to avoid modifying the original system
        system_copy = deepcopy(system)
        
        # Set power flow settings
        system_copy.PFlow.init()
        system_copy.PFlow.config.flat_start = 0
        system_copy.PFlow.config.max_iter = 30
        system_copy.PFlow.config.tol = 1e-6
        
        # Run power flow
        system_copy.PFlow.run()
        
        # Check convergence
        converged = system_copy.PFlow.converged
        
        # Get results
        results = {
            'converged': converged,
            'iterations': system_copy.PFlow.niter,
            'error': system_copy.PFlow.mis,
            'elapsed_time': system_copy.PFlow.t_total
        }
        
        if converged:
            logger.info(f"Power flow converged in {results['iterations']} iterations")
        else:
            logger.warning(f"Power flow did not converge after {results['iterations']} iterations")
        
        return converged, results, system_copy
    
    except Exception as e:
        logger.error(f"Failed to run power flow: {str(e)}")
        raise

def calculate_line_loadings(system, mva_base=100.0):
    """
    Calculate loading percentage for all lines in the system.
    
    Args:
        system (andes.System): ANDES system with solved power flow
        mva_base (float): System MVA base
        
    Returns:
        pd.DataFrame: DataFrame with line loading information
    """
    try:
        # Check if power flow was run
        if not hasattr(system, 'f') or not hasattr(system.f, 'Line'):
            raise ValueError("Power flow solution not available. Run power flow first.")
        
        line_data = []
        
        # Calculate complex power flow for each line
        for i in range(system.Line.n):
            from_bus = system.Line.bus1[i]
            to_bus = system.Line.bus2[i]
            
            # Get line flow in complex power
            s_from = complex(system.f.Line.pf[i], system.f.Line.qf[i]) * mva_base
            s_to = complex(system.f.Line.pt[i], system.f.Line.qt[i]) * mva_base
            
            # Calculate apparent power magnitude at both ends
            s_from_mag = abs(s_from)
            s_to_mag = abs(s_to)
            s_max = max(s_from_mag, s_to_mag)
            
            # Calculate loading percentage
            rating = system.Line.rate_a[i] * mva_base
            if rating > 0:
                loading_percent = 100 * s_max / rating
            else:
                loading_percent = 0  # No rating specified
            
            line_name = system.Line.name[i] if system.Line.name[i] else f"Line_{i}"
            
            line_data.append({
                'line_idx': i,
                'line_name': line_name,
                'from_bus': from_bus,
                'to_bus': to_bus,
                'p_from_mw': system.f.Line.pf[i] * mva_base,
                'q_from_mvar': system.f.Line.qf[i] * mva_base,
                's_from_mva': s_from_mag,
                'p_to_mw': system.f.Line.pt[i] * mva_base,
                'q_to_mvar': system.f.Line.qt[i] * mva_base,
                's_to_mva': s_to_mag,
                's_max_mva': s_max,
                'rating_mva': rating,
                'loading_percent': loading_percent
            })
        
        return pd.DataFrame(line_data)
    
    except Exception as e:
        logger.error(f"Failed to calculate line loadings: {str(e)}")
        raise

def calculate_overload_index(system, p=2, weight_factor=1.0, mva_base=100.0):
    """
    Calculate the overload performance index as defined in the equation:
    f_x = sum(wf_i * (S_mean,i / S_max,i)^p) for all lines
    
    Args:
        system (andes.System): ANDES system with solved power flow
        p (int): Power factor in the overload index formula
        weight_factor (float): Weight factor for all lines
        mva_base (float): System MVA base
        
    Returns:
        float: Overload performance index
        pd.DataFrame: DataFrame with line-wise overload indices
    """
    try:
        # Get line loadings
        line_loadings = calculate_line_loadings(system, mva_base)
        
        # Calculate mean apparent power for each line
        line_loadings['s_mean_mva'] = (line_loadings['s_from_mva'] + line_loadings['s_to_mva']) / 2
        
        # Calculate line-wise indices
        line_indices = []
        total_index = 0.0
        
        for _, line in line_loadings.iterrows():
            if line['rating_mva'] > 0:
                # Calculate the term (S_mean / S_max)^p
                term = (line['s_mean_mva'] / line['rating_mva']) ** p
                # Apply weight factor
                weighted_term = weight_factor * term
                
                line_indices.append({
                    'line_idx': line['line_idx'],
                    'line_name': line['line_name'],
                    's_mean_mva': line['s_mean_mva'],
                    'rating_mva': line['rating_mva'],
                    'term': term,
                    'weighted_term': weighted_term
                })
                
                total_index += weighted_term
        
        line_indices_df = pd.DataFrame(line_indices)
        
        logger.info(f"Calculated overload index: {total_index:.4f}")
        return total_index, line_indices_df
    
    except Exception as e:
        logger.error(f"Failed to calculate overload index: {str(e)}")
        raise

def check_voltage_violations(system, v_min=0.95, v_max=1.05):
    """
    Check for bus voltage violations.
    
    Args:
        system (andes.System): ANDES system with solved power flow
        v_min (float): Minimum acceptable voltage in p.u.
        v_max (float): Maximum acceptable voltage in p.u.
        
    Returns:
        pd.DataFrame: DataFrame with voltage violation information
    """
    try:
        # Check if power flow was run
        if not hasattr(system, 'f') or not hasattr(system.f, 'Bus'):
            raise ValueError("Power flow solution not available. Run power flow first.")
        
        violations = []
        
        for i in range(system.Bus.n):
            v_magnitude = abs(system.f.Bus.v[i])
            
            if v_magnitude < v_min or v_magnitude > v_max:
                bus_name = system.Bus.name[i] if system.Bus.name[i] else f"Bus_{i}"
                violations.append({
                    'bus_idx': i,
                    'bus_name': bus_name,
                    'voltage_pu': v_magnitude,
                    'violation_type': 'Low' if v_magnitude < v_min else 'High',
                    'deviation': abs(v_magnitude - (v_min if v_magnitude < v_min else v_max))
                })
        
        violations_df = pd.DataFrame(violations)
        
        if len(violations) > 0:
            logger.warning(f"Found {len(violations)} voltage violations")
        else:
            logger.info("No voltage violations found")
        
        return violations_df
    
    except Exception as e:
        logger.error(f"Failed to check voltage violations: {str(e)}")
        raise

def perform_static_security_assessment(system, line_outages=None, v_min=0.95, v_max=1.05, p=2):
    """
    Perform comprehensive static security assessment including N-1 contingency analysis.
    
    Args:
        system (andes.System): ANDES system object
        line_outages (list, optional): List of line indices to analyze as contingencies
                                       If None, all lines will be considered
        v_min (float): Minimum acceptable voltage in p.u.
        v_max (float): Maximum acceptable voltage in p.u.
        p (int): Power factor in the overload index formula
        
    Returns:
        dict: Dictionary with assessment results
    """
    try:
        # Run base case power flow
        logger.info("Running base case power flow")
        base_converged, base_results, base_solved = run_power_flow(system)
        
        if not base_converged:
            logger.error("Base case power flow did not converge")
            return {
                'base_case_converged': False,
                'overall_secure': False,
                'reason': 'Base case power flow did not converge'
            }
        
        # Calculate base case indices
        base_loadings = calculate_line_loadings(base_solved)
        base_overload_idx, base_overload_details = calculate_overload_index(base_solved, p=p)
        base_voltage_violations = check_voltage_violations(base_solved, v_min, v_max)
        
        # Determine if line outages were specified or use all lines
        if line_outages is None:
            line_outages = list(range(system.Line.n))
        
        # Perform N-1 contingency analysis
        logger.info(f"Performing N-1 contingency analysis for {len(line_outages)} line outages")
        
        contingency_results = []
        
        for line_idx in line_outages:
            # Create a modified system with the line outage
            from topology_changes import disconnect_line
            
            logger.info(f"Analyzing contingency: Line outage {line_idx}")
            contingency_system = disconnect_line(system, line_idx)
            
            # Run power flow for this contingency
            cont_converged, cont_results, cont_solved = run_power_flow(contingency_system)
            
            if cont_converged:
                # Calculate indices for this contingency
                cont_loadings = calculate_line_loadings(cont_solved)
                cont_overload_idx, _ = calculate_overload_index(cont_solved, p=p)
                cont_voltage_violations = check_voltage_violations(cont_solved, v_min, v_max)
                
                # Check for overloaded lines (>100% loading)
                overloaded_lines = cont_loadings[cont_loadings['loading_percent'] > 100]
                
                contingency_results.append({
                    'contingency_type': 'line_outage',
                    'element_idx': line_idx,
                    'converged': cont_converged,
                    'overload_index': cont_overload_idx,
                    'voltage_violations': len(cont_voltage_violations),
                    'overloaded_lines': len(overloaded_lines),
                    'secure': len(overloaded_lines) == 0 and len(cont_voltage_violations) == 0
                })
            else:
                contingency_results.append({
                    'contingency_type': 'line_outage',
                    'element_idx': line_idx,
                    'converged': cont_converged,
                    'secure': False,
                    'reason': 'Power flow did not converge'
                })
        
        # Compile overall security assessment
        contingency_df = pd.DataFrame(contingency_results)
        secure_contingencies = contingency_df[contingency_df['secure'] == True]
        
        # System is secure if all contingencies are secure
        overall_secure = len(secure_contingencies) == len(contingency_df)
        
        assessment_results = {
            'base_case_converged': base_converged,
            'base_overload_index': base_overload_idx,
            'base_voltage_violations': len(base_voltage_violations),
            'overall_secure': overall_secure,
            'total_contingencies': len(contingency_df),
            'secure_contingencies': len(secure_contingencies),
            'contingency_details': contingency_df.to_dict('records'),
            'base_case_line_loadings': base_loadings.to_dict('records'),
            'base_case_voltage_violations': base_voltage_violations.to_dict('records') if len(base_voltage_violations) > 0 else []
        }
        
        logger.info(f"Static security assessment complete. System is {'secure' if overall_secure else 'insecure'}")
        return assessment_results
    
    except Exception as e:
        logger.error(f"Failed to perform static security assessment: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    import load_ieee_system
    
    # Load IEEE 68-bus system
    system = load_ieee_system.load_ieee68()
    
    # Run static security assessment for a few line outages
    line_outages = list(range(5))  # First 5 lines
    assessment = perform_static_security_assessment(system, line_outages, v_min=0.9, v_max=1.1)
    
    # Print summary
    print("\nStatic Security Assessment Summary:")
    print(f"Base case converged: {assessment['base_case_converged']}")
    print(f"Base overload index: {assessment['base_overload_index']:.4f}")
    print(f"Base voltage violations: {assessment['base_voltage_violations']}")
    print(f"System secure under N-1 contingencies: {assessment['overall_secure']}")
    print(f"Total contingencies analyzed: {assessment['total_contingencies']}")
    print(f"Secure contingencies: {assessment['secure_contingencies']}")