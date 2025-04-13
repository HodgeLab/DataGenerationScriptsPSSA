"""
Module for performing static security assessment through power flows in Dynawo.
"""

import dynawo
import numpy as np
import pandas as pd


class StaticSecurityAssessor:
    """Class to perform static security assessment in Dynawo power system models."""
    
    def __init__(self, simulation):
        """
        Initialize the static security assessor.
        
        Args:
            simulation: A Dynawo simulation object
        """
        self.simulation = simulation
        self.network = simulation.get_network()
        self.results = None
    
    def run_power_flow(self):
        """
        Run a power flow analysis on the system.
        
        Returns:
            True if the power flow converged, False otherwise
        """
        print("Running power flow analysis...")
        
        # Create a power flow solver
        pf_solver = dynawo.PowerFlowSolver(self.network)
        
        # Set power flow options
        pf_solver.set_max_iterations(20)
        pf_solver.set_tolerance(1e-6)
        
        # Run the power flow
        converged = pf_solver.solve()
        
        if converged:
            print("Power flow converged successfully")
        else:
            print("Power flow did not converge")
        
        return converged
    
    def calculate_overload_index(self, p=2, weights=None):
        """
        Calculate the overload index as per equation:
        f_x = sum(w_i * (S_mean,i / S_max,i)^p)
        
        Args:
            p: Exponent for the index calculation (default: 2)
            weights: Optional weights for each line (default: equal weights)
            
        Returns:
            The calculated overload index value
        """
        print("Calculating overload index...")
        
        # Run power flow if not already done
        if not hasattr(self, '_pf_run') or not self._pf_run:
            converged = self.run_power_flow()
            if not converged:
                print("Cannot calculate overload index - power flow did not converge")
                return None
            self._pf_run = True
        
        # Get all lines
        lines = list(self.network.get_lines())
        n_lines = len(lines)
        
        # Set default weights if not provided
        if weights is None:
            weights = np.ones(n_lines) / n_lines
        elif len(weights) != n_lines:
            print(f"Warning: Number of weights ({len(weights)}) does not match number of lines ({n_lines})")
            weights = np.ones(n_lines) / n_lines
        
        # Calculate the overload index
        overload_index = 0.0
        line_indices = []
        
        for i, line in enumerate(lines):
            if not line.is_connected():
                continue
                
            # Get apparent power flow and rating
            p_from = line.get_p_from()
            q_from = line.get_q_from()
            s_mean = np.sqrt(p_from**2 + q_from**2)
            s_max = line.get_current_limit() * line.get_v_from()  # S = V * I
            
            if s_max <= 0:
                print(f"Warning: Line {line.get_id()} has zero or negative rating. Skipping.")
                continue
            
            # Calculate term for this line
            line_term = weights[i] * (s_mean / s_max) ** p
            overload_index += line_term
            
            # Store individual line index
            line_indices.append({
                'line_id': line.get_id(),
                'from_bus': line.get_bus1_id(),
                'to_bus': line.get_bus2_id(),
                's_mean': s_mean,
                's_max': s_max,
                'loading_percent': (s_mean / s_max) * 100,
                'term_value': line_term
            })
        
        # Store results
        self.results = {
            'overload_index': overload_index,
            'line_indices': pd.DataFrame(line_indices).sort_values(by='loading_percent', ascending=False)
        }
        
        print(f"Overload index: {overload_index:.4f}")
        return overload_index
    
    def identify_critical_lines(self, threshold_percent=90):
        """
        Identify lines that are critically loaded.
        
        Args:
            threshold_percent: Loading percentage threshold to consider a line critical
            
        Returns:
            DataFrame containing the critical lines
        """
        if self.results is None:
            self.calculate_overload_index()
        
        critical_lines = self.results['line_indices'][
            self.results['line_indices']['loading_percent'] >= threshold_percent
        ]
        
        print(f"Found {len(critical_lines)} critical lines loaded above {threshold_percent}%")
        return critical_lines
    
    def check_voltage_violations(self, v_min=0.95, v_max=1.05):
        """
        Check for voltage violations in the system.
        
        Args:
            v_min: Minimum acceptable voltage in p.u.
            v_max: Maximum acceptable voltage in p.u.
            
        Returns:
            DataFrame containing buses with voltage violations
        """
        # Run power flow if not already done
        if not hasattr(self, '_pf_run') or not self._pf_run:
            converged = self.run_power_flow()
            if not converged:
                print("Cannot check voltage violations - power flow did not converge")
                return None
            self._pf_run = True
        
        violations = []
        
        for bus in self.network.get_buses():
            if not bus.is_connected():
                continue
                
            v = bus.get_v() / bus.get_v_nom()  # Convert to p.u.
            
            if v < v_min or v > v_max:
                violations.append({
                    'bus_id': bus.get_id(),
                    'voltage_pu': v,
                    'violation': 'Low' if v < v_min else 'High',
                    'deviation': min(abs(v - v_min), abs(v - v_max))
                })
        
        df_violations = pd.DataFrame(violations).sort_values(by='deviation', ascending=False)
        
        print(f"Found {len(df_violations)} buses with voltage violations")
        return df_violations
    
    def get_branch_flows(self):
        """
        Get power flows on all branches.
        
        Returns:
            DataFrame containing branch flow information
        """
        # Run power flow if not already done
        if not hasattr(self, '_pf_run') or not self._pf_run:
            converged = self.run_power_flow()
            if not converged:
                print("Cannot get branch flows - power flow did not converge")
                return None
            self._pf_run = True
        
        branch_flows = []
        
        # Get line flows
        for line in self.network.get_lines():
            if not line.is_connected():
                continue
                
            p_from = line.get_p_from()
            q_from = line.get_q_from()
            s_from = np.sqrt(p_from**2 + q_from**2)
            
            p_to = line.get_p_to()
            q_to = line.get_q_to()
            s_to = np.sqrt(p_to**2 + q_to**2)
            
            s_max = line.get_current_limit() * line.get_v_from()
            loading = (max(s_from, s_to) / s_max) * 100 if s_max > 0 else 0
            
            branch_flows.append({
                'branch_id': line.get_id(),
                'type': 'Line',
                'from_bus': line.get_bus1_id(),
                'to_bus': line.get_bus2_id(),
                'p_from_mw': p_from,
                'q_from_mvar': q_from,
                's_from_mva': s_from,
                'p_to_mw': p_to,
                'q_to_mvar': q_to,
                's_to_mva': s_to,
                's_max_mva': s_max,
                'loading_percent': loading
            })
        
        # Get transformer flows
        for transformer in self.network.get_transformers():
            if not transformer.is_connected():
                continue
                
            p_from = transformer.get_p_from()
            q_from = transformer.get_q_from()
            s_from = np.sqrt(p_from**2 + q_from**2)
            
            p_to = transformer.get_p_to()
            q_to = transformer.get_q_to()
            s_to = np.sqrt(p_to**2 + q_to**2)
            
            s_max = transformer.get_current_limit() * transformer.get_v_from() if hasattr(transformer, 'get_current_limit') else 0
            loading = (max(s_from, s_to) / s_max) * 100 if s_max > 0 else 0
            
            branch_flows.append({
                'branch_id': transformer.get_id(),
                'type': 'Transformer',
                'from_bus': transformer.get_bus1_id(),
                'to_bus': transformer.get_bus2_id(),
                'p_from_mw': p_from,
                'q_from_mvar': q_from,
                's_from_mva': s_from,
                'p_to_mw': p_to,
                'q_to_mvar': q_to,
                's_to_mva': s_to,
                's_max_mva': s_max,
                'loading_percent': loading
            })
        
        return pd.DataFrame(branch_flows).sort_values(by='loading_percent', ascending=False)
    
    def get_bus_voltages(self):
        """
        Get voltages at all buses.
        
        Returns:
            DataFrame containing bus voltage information
        """
        # Run power flow if not already done
        if not hasattr(self, '_pf_run') or not self._pf_run:
            converged = self.run_power_flow()
            if not converged:
                print("Cannot get bus voltages - power flow did not converge")
                return None
            self._pf_run = True
        
        bus_voltages = []
        
        for bus in self.network.get_buses():
            if not bus.is_connected():
                continue
                
            v = bus.get_v()
            v_nom = bus.get_v_nom()
            v_pu = v / v_nom
            angle = bus.get_angle()
            
            bus_voltages.append({
                'bus_id': bus.get_id(),
                'voltage_kv': v,
                'voltage_pu': v_pu,
                'angle_deg': angle,
                'v_nom_kv': v_nom
            })
        
        return pd.DataFrame(bus_voltages).sort_values(by='voltage_pu')
    
    def assess_static_security(self, overload_threshold=90, v_min=0.95, v_max=1.05):
        """
        Perform a complete static security assessment.
        
        Args:
            overload_threshold: Line loading percentage threshold
            v_min: Minimum acceptable voltage in p.u.
            v_max: Maximum acceptable voltage in p.u.
            
        Returns:
            Dictionary containing assessment results
        """
        print("Performing static security assessment...")
        
        # Run power flow
        converged = self.run_power_flow()
        if not converged:
            return {'converged': False, 'secure': False, 'message': 'Power flow did not converge'}
        
        # Calculate overload index
        overload_index = self.calculate_overload_index()
        
        # Check for critical lines
        critical_lines = self.identify_critical_lines(overload_threshold)
        
        # Check for voltage violations
        voltage_violations = self.check_voltage_violations(v_min, v_max)
        
        # Determine if the system is secure
        is_secure = (len(critical_lines) == 0) and (len(voltage_violations) == 0)
        
        # Compile results
        assessment = {
            'converged': converged,
            'secure': is_secure,
            'overload_index': overload_index,
            'critical_lines': critical_lines,
            'voltage_violations': voltage_violations,
            'branch_flows': self.get_branch_flows(),
            'bus_voltages': self.get_bus_voltages()
        }
        
        if is_secure:
            print("System is statically secure")
        else:
            print("System is statically insecure")
            if len(critical_lines) > 0:
                print(f"  - {len(critical_lines)} lines loaded above {overload_threshold}%")
            if len(voltage_violations) > 0:
                print(f"  - {len(voltage_violations)} buses with voltage violations")
        
        return assessment


# Example usage
if __name__ == "__main__":
    # This requires a simulation object to be created first
    # from load_ieee_systems import IEEESystemLoader
    # loader = IEEESystemLoader()
    # sim = loader.load_ieee68()
    # assessor = StaticSecurityAssessor(sim)
    # results = assessor.assess_static_security()
    pass