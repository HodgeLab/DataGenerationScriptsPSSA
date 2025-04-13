"""
Module for performing voltage stability assessment in Dynawo.
"""

import dynawo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class VoltageStabilityAnalyzer:
    """Class to perform voltage stability analysis in Dynawo power system models."""
    
    def __init__(self, simulation):
        """
        Initialize the voltage stability analyzer.
        
        Args:
            simulation: A Dynawo simulation object
        """
        self.simulation = simulation
        self.network = simulation.get_network()
        self.results = None
    
    def run_continuation_power_flow(self, loading_parameter='load_p', start_value=1.0, 
                                   max_value=2.0, step_size=0.05):
        """
        Run a continuation power flow to determine voltage collapse point.
        
        Args:
            loading_parameter: Parameter to increase ('load_p', 'load_q', or 'both')
            start_value: Starting value of the loading parameter
            max_value: Maximum value to try
            step_size: Step size for increasing the loading parameter
            
        Returns:
            DataFrame with continuation power flow results
        """
        print(f"Running continuation power flow (increasing {loading_parameter})...")
        
        # Store original load values
        original_loads = {}
        for load in self.network.get_loads():
            load_id = load.get_id()
            original_loads[load_id] = {
                'p': load.get_p(),
                'q': load.get_q()
            }
        
        results = []
        current_value = start_value
        last_converged = None
        
        while current_value <= max_value:
            # Scale loads
            for load_id, orig in original_loads.items():
                load = self.network.get_load(load_id)
                
                if loading_parameter == 'load_p' or loading_parameter == 'both':
                    load.set_p(orig['p'] * current_value)
                
                if loading_parameter == 'load_q' or loading_parameter == 'both':
                    load.set_q(orig['q'] * current_value)
            
            # Run power flow
            pf_solver = dynawo.PowerFlowSolver(self.network)
            pf_solver.set_max_iterations(50)  # Increase iterations for stressed conditions
            converged = pf_solver.solve()
            
            # Store results
            iteration_results = {
                'loading_parameter': current_value,
                'converged': converged
            }
            
            # If converged, store bus voltages
            if converged:
                last_converged = current_value
                min_voltage = float('inf')
                max_voltage = float('-inf')
                
                # Get min and max voltages
                for bus in self.network.get_buses():
                    if not bus.is_connected():
                        continue
                    
                    v_pu = bus.get_v() / bus.get_v_nom()
                    min_voltage = min(min_voltage, v_pu)
                    max_voltage = max(max_voltage, v_pu)
                
                iteration_results['min_voltage_pu'] = min_voltage
                iteration_results['max_voltage_pu'] = max_voltage
            else:
                print(f"Power flow did not converge at loading = {current_value}")
                if last_converged:
                    print(f"Last convergence was at loading = {last_converged}")
                break
            
            results.append(iteration_results)
            current_value += step_size
        
        # Restore original load values
        for load_id, orig in original_loads.items():
            load = self.network.get_load(load_id)
            load.set_p(orig['p'])
            load.set_q(orig['q'])
        
        # Run one more power flow to restore the base case
        pf_solver = dynawo.PowerFlowSolver(self.network)
        pf_solver.solve()
        
        df_results = pd.DataFrame(results)
        
        if last_converged:
            print(f"Maximum loading parameter value: {last_converged}")
            
            # Calculate loading margin
            loading_margin = last_converged - start_value
            print(f"Loading margin: {loading_margin:.2f} ({loading_margin*100:.1f}%)")
        else:
            print("All power flows converged up to maximum loading")
        
        return df_results
    
    def calculate_pv_curves(self, load_bus_ids, generator_bus_ids=None, max_loading=2.0, step_size=0.05):
        """
        Calculate P-V curves for specified load buses.
        
        Args:
            load_bus_ids: List of load bus IDs to monitor
            generator_bus_ids: List of generator bus IDs to scale generation
            max_loading: Maximum loading multiplier
            step_size: Loading step size
            
        Returns:
            Dictionary containing P-V curve data
        """
        print("Calculating P-V curves...")
        
        # Store original values
        original_loads = {}
        for load in self.network.get_loads():
            load_id = load.get_id()
            bus_id = load.get_bus_id()
            if bus_id in load_bus_ids:
                original_loads[load_id] = {
                    'bus_id': bus_id,
                    'p': load.get_p(),
                    'q': load.get_q()
                }
        
        original_gens = {}
        if generator_bus_ids:
            for gen in self.network.get_generators():
                gen_id = gen.get_id()
                bus_id = gen.get_bus_id()
                if bus_id in generator_bus_ids:
                    original_gens[gen_id] = {
                        'bus_id': bus_id,
                        'p': gen.get_p()
                    }
        
        # Initialize results
        pv_curves = {bus_id: {'loading': [], 'voltage': []} for bus_id in load_bus_ids}
        
        # Run continuation power flow
        current_loading = 1.0
        last_converged = None
        
        while current_loading <= max_loading:
            # Scale loads
            for load_id, data in original_loads.items():
                load = self.network.get_load(load_id)
                load.set_p(data['p'] * current_loading)
                load.set_q(data['q'] * current_loading)
            
            # Scale generators if specified
            if generator_bus_ids:
                total_orig_p = sum(data['p'] for data in original_loads.values())
                total_new_p = total_orig_p * current_loading
                p_increase = total_new_p - total_orig_p
                
                if p_increase > 0:
                    # Distribute increase proportionally among generators
                    total_gen_p = sum(data['p'] for data in original_gens.values())
                    for gen_id, data in original_gens.items():
                        gen = self.network.get_generator(gen_id)
                        p_share = data['p'] / total_gen_p if total_gen_p > 0 else 1.0 / len(original_gens)
                        gen.set_p(data['p'] + p_increase * p_share)
            
            # Run power flow
            pf_solver = dynawo.PowerFlowSolver(self.network)
            pf_solver.set_max_iterations(50)
            converged = pf_solver.solve()
            
            if converged:
                last_converged = current_loading
                
                # Store voltage for each monitored bus
                for bus_id in load_bus_ids:
                    bus = self.network.get_bus(bus_id)
                    v_pu = bus.get_v() / bus.get_v_nom()
                    
                    pv_curves[bus_id]['loading'].append(current_loading)
                    pv_curves[bus_id]['voltage'].append(v_pu)
            else:
                print(f"Power flow did not converge at loading = {current_loading}")
                if last_converged:
                    print(f"Last convergence was at loading = {last_converged}")
                break
            
            current_loading += step_size
        
        # Restore original values
        for load_id, data in original_loads.items():
            load = self.network.get_load(load_id)
            load.set_p(data['p'])
            load.set_q(data['q'])
        
        if generator_bus_ids:
            for gen_id, data in original_gens.items():
                gen = self.network.get_generator(gen_id)
                gen.set_p(data['p'])
        
        # Restore base case
        pf_solver = dynawo.PowerFlowSolver(self.network)
        pf_solver.solve()
        
        if last_converged:
            print(f"Maximum loading: {last_converged}")
        
        return pv_curves
    
    def calculate_vq_curves(self, bus_ids, q_min=-100, q_max=100, step_size=5):
        """
        Calculate V-Q curves for specified buses.
        
        Args:
            bus_ids: List of bus IDs to analyze
            q_min: Minimum reactive power (MVAR)
            q_max: Maximum reactive power (MVAR)
            step_size: Step size for reactive power
            
        Returns:
            Dictionary containing V-Q curve data
        """
        print("Calculating V-Q curves...")
        
        # Create result structure
        vq_curves = {bus_id: {'q': [], 'voltage': []} for bus_id in bus_ids}
        
        # Save original network state to restore later
        original_state = {}
        for bus_id in bus_ids:
            # Check if there's already a generator at this bus
            gen_at_bus = None
            for gen in self.network.get_generators():
                if gen.get_bus_id() == bus_id:
                    gen_at_bus = gen.get_id()
                    original_state[gen_at_bus] = {
                        'p': gen.get_p(),
                        'q': gen.get_q(),
                        'v_setpoint': gen.get_v_regul_setpoint() if hasattr(gen, 'get_v_regul_setpoint') else None
                    }
                    break
            
            # If no generator, we'll need to add a temporary one
            if not gen_at_bus:
                original_state[bus_id] = {'had_generator': False}
        
        # Process each bus
        for bus_id in bus_ids:
            print(f"Processing bus {bus_id}...")
            
            # Find if there's a generator at this bus
            gen_at_bus = None
            for gen in self.network.get_generators():
                if gen.get_bus_id() == bus_id:
                    gen_at_bus = gen.get_id()
                    break
            
            # If no generator, add a temporary one with P=0
            temp_gen_id = None
            if not gen_at_bus:
                bus = self.network.get_bus(bus_id)
                temp_gen_id = f"TEMP_GEN_{bus_id}"
                temp_gen = self.network.create_generator(
                    id=temp_gen_id,
                    bus_id=bus_id,
                    p=0,
                    q=0,
                    v_regul=False
                )
                gen_at_bus = temp_gen_id
            
            # Set generator to various Q values and measure voltage
            q_value = q_min
            while q_value <= q_max:
                # Set reactive power
                gen = self.network.get_generator(gen_at_bus)
                gen.set_q(q_value)
                
                # Disable voltage control for this test
                if hasattr(gen, 'set_v_regul'):
                    gen.set_v_regul(False)
                
                # Run power flow
                pf_solver = dynawo.PowerFlowSolver(self.network)
                converged = pf_solver.solve()
                
                if converged:
                    # Get voltage
                    bus = self.network.get_bus(bus_id)
                    v_pu = bus.get_v() / bus.get_v_nom()
                    
                    # Store results
                    vq_curves[bus_id]['q'].append(q_value)
                    vq_curves[bus_id]['voltage'].append(v_pu)
                else:
                    print(f"Power flow did not converge at Q = {q_value} for bus {bus_id}")
                
                q_value += step_size
            
            # Remove temporary generator if we added one
            if temp_gen_id:
                self.network.remove_generator(temp_gen_id)
        
        # Restore original network state
        for id, data in original_state.items():
            if 'had_generator' in data and not data['had_generator']:
                # We already removed the temporary generator
                continue
            
            gen = self.network.get_generator(id)
            if gen:
                gen.set_p(data['p'])
                gen.set_q(data['q'])
                if data['v_setpoint'] is not None and hasattr(gen, 'set_v_regul_setpoint'):
                    gen.set_v_regul_setpoint(data['v_setpoint'])
                    gen.set_v_regul(True)
        
        # Restore base case
        pf_solver = dynawo.PowerFlowSolver(self.network)
        pf_solver.solve()
        
        return vq_curves
    
    def run_time_domain_simulation(self, duration=20.0, time_step=0.01):
        """
        Run a time-domain simulation to assess voltage stability.
        
        Args:
            duration: Simulation duration (seconds)
            time_step: Simulation time step (seconds)
            
        Returns:
            Dictionary containing simulation results
        """
        print("Running time-domain simulation for voltage stability assessment...")
        
        # Set simulation parameters
        self.simulation.set_duration(duration)
        self.simulation.set_time_step(time_step)
        
        # Run the simulation
        result = self.simulation.run()
        
        if result:
            print("Time-domain simulation completed successfully")
            
            # Extract bus voltages
            buses = self.network.get_buses()
            bus_voltages = {}
            
            for bus in buses:
                bus_id = bus.get_id()
                if bus.is_connected():
                    v_nom = bus.get_v_nom()
                    voltages = np.array(bus.get_variable_values('v'))
                    voltages_pu = voltages / v_nom
                    
                    bus_voltages[bus_id] = {
                        'time': np.linspace(0, duration, len(voltages)),
                        'voltage': voltages,
                        'voltage_pu': voltages_pu
                    }
            
            return {
                'success': True,
                'bus_voltages': bus_voltages
            }
        else:
            print("Time-domain simulation failed")
            return {'success': False}
    
    def check_voltage_criteria(self, simulation_results, v_min=0.8, v_max=1.1, violation_duration=0.5, 
                              time_step=0.01):
        """
        Check if voltage criteria are met in the time-domain simulation.
        
        Args:
            simulation_results: Results from time-domain simulation
            v_min: Minimum acceptable voltage in p.u.
            v_max: Maximum acceptable voltage in p.u.
            violation_duration: Maximum acceptable duration for voltage violations (seconds)
            time_step: Simulation time step (seconds)
            
        Returns:
            Dictionary containing voltage violation assessment
        """
        if not simulation_results['success']:
            return {'success': False, 'message': 'Simulation failed'}
        
        bus_voltages = simulation_results['bus_voltages']
        
        # Maximum allowed consecutive violations
        max_violations = int(violation_duration / time_step)
        
        violations = {}
        for bus_id, data in bus_voltages.items():
            voltages_pu = data['voltage_pu']
            
            # Check for violations
            low_violations = voltages_pu < v_min
            high_violations = voltages_pu > v_max
            
            # Count consecutive violations
            max_consecutive_low = self._count_max_consecutive(low_violations)
            max_consecutive_high = self._count_max_consecutive(high_violations)
            
            max_consecutive = max(max_consecutive_low, max_consecutive_high)
            max_duration = max_consecutive * time_step
            
            if max_consecutive > max_violations:
                violation_type = "Low" if max_consecutive_low > max_consecutive_high else "High"
                violations[bus_id] = {
                    'bus_id': bus_id,
                    'violation_type': violation_type,
                    'max_consecutive': max_consecutive,
                    'duration': max_duration,
                    'min_voltage': np.min(voltages_pu),
                    'max_voltage': np.max(voltages_pu)
                }
        
        is_secure = len(violations) == 0
        
        if is_secure:
            print("Voltage stability criteria met (no sustained violations)")
        else:
            print(f"Voltage stability criteria not met ({len(violations)} buses with sustained violations)")
        
        return {
            'success': True,
            'secure': is_secure,
            'violations': violations,
            'criteria': {
                'v_min': v_min,
                'v_max': v_max,
                'violation_duration': violation_duration
            }
        }
    
    def _count_max_consecutive(self, boolean_array):
        """
        Count maximum consecutive True values in a boolean array.
        
        Args:
            boolean_array: NumPy array of booleans
            
        Returns:
            Maximum count of consecutive True values
        """
        if not np.any(boolean_array):
            return 0
    
    def assess_voltage_stability(self, v_min=0.8, v_max=1.1, violation_duration=0.5):
        """
        Perform a complete voltage stability assessment using time-domain simulation.
        
        Args:
            v_min: Minimum acceptable voltage in p.u.
            v_max: Maximum acceptable voltage in p.u.
            violation_duration: Maximum acceptable duration for voltage violations (seconds)
            
        Returns:
            Dictionary containing assessment results
        """
        print("Performing voltage stability assessment...")
        
        # Run time-domain simulation
        simulation_results = self.run_time_domain_simulation()
        
        if not simulation_results['success']:
            return {'success': False, 'secure': False, 'message': 'Simulation failed'}
        
        # Check voltage criteria
        criteria_results = self.check_voltage_criteria(
            simulation_results, 
            v_min=v_min, 
            v_max=v_max, 
            violation_duration=violation_duration
        )
        
        # Compile results
        assessment = {
            'success': True,
            'secure': criteria_results['secure'],
            'violations': criteria_results['violations'],
            'criteria': criteria_results['criteria'],
            'simulation_results': simulation_results
        }
        
        return assessment


# Example usage
if __name__ == "__main__":
    # This requires a simulation object to be created first
    # from load_ieee_systems import IEEESystemLoader
    # loader = IEEESystemLoader()
    # sim = loader.load_ieee68()
    # analyzer = VoltageStabilityAnalyzer(sim)
    # results = analyzer.assess_voltage_stability()
    pass
        
        # # Add False at both ends to handle boundary cases
        # padded = np.concatenate(([False], boolean_array, [False]))
        
        # # Find where values change
        # diff = np.diff(padded.astype(int))
        
        # # Start indices of runs of True
        # run_starts = np.where(diff > 0)[0]
        
        # # End indices of runs of True
        # run_ends = np.where(diff < 0)[0]
        
        # # Calculate lengths of runs
        # run_lengths = run_ends - run_starts
        
        #     return np.max(run_lengths)
