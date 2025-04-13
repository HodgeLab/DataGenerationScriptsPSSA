"""
Module for performing transient stability assessment in Dynawo.
"""

import dynawo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TransientStabilityAnalyzer:
    """Class to perform transient stability analysis in Dynawo power system models."""
    
    def __init__(self, simulation):
        """
        Initialize the transient stability analyzer.
        
        Args:
            simulation: A Dynawo simulation object
        """
        self.simulation = simulation
        self.network = simulation.get_network()
        self.results = None
    
    def apply_fault(self, fault_type, element_id, start_time=1.0, duration=0.1, **kwargs):
        """
        Apply a fault to the system.
        
        Args:
            fault_type: Type of fault ('bus_fault', 'line_trip', 'generator_trip', etc.)
            element_id: ID of the element to apply fault to
            start_time: Fault start time (seconds)
            duration: Fault duration (seconds)
            **kwargs: Additional fault parameters
            
        Returns:
            True if fault was applied successfully, False otherwise
        """
        print(f"Applying {fault_type} to {element_id} at t={start_time}s for {duration}s")
        
        try:
            # Create a contingency
            contingency = dynawo.Contingency(self.simulation)
            
            if fault_type == 'bus_fault':
                # Bus fault (three-phase short circuit)
                bus = self.network.get_bus(element_id)
                if not bus:
                    print(f"Bus {element_id} not found")
                    return False
                
                resistance = kwargs.get('resistance', 0.0)
                contingency.add_bus_fault(
                    bus_id=element_id,
                    start_time=start_time,
                    end_time=start_time + duration,
                    resistance=resistance
                )
                
            elif fault_type == 'line_trip':
                # Line trip
                line = self.network.get_line(element_id)
                if not line:
                    print(f"Line {element_id} not found")
                    return False
                
                contingency.add_line_trip(
                    line_id=element_id,
                    start_time=start_time,
                    end_time=None  # Permanent trip
                )
                
            elif fault_type == 'line_fault':
                # Line fault followed by trip
                line = self.network.get_line(element_id)
                if not line:
                    print(f"Line {element_id} not found")
                    return False
                
                location = kwargs.get('location', 50.0)  # % from bus1, default to middle
                resistance = kwargs.get('resistance', 0.0)
                permanent = kwargs.get('permanent', True)
                trip_time = kwargs.get('trip_time', start_time + duration)
                
                # Add fault
                contingency.add_line_fault(
                    line_id=element_id,
                    start_time=start_time,
                    end_time=start_time + duration,
                    location=location,
                    resistance=resistance
                )
                
                # Add line trip if permanent
                if permanent:
                    contingency.add_line_trip(
                        line_id=element_id,
                        start_time=trip_time,
                        end_time=None
                    )
                
            elif fault_type == 'generator_trip':
                # Generator trip
                gen = self.network.get_generator(element_id)
                if not gen:
                    print(f"Generator {element_id} not found")
                    return False
                
                contingency.add_generator_trip(
                    generator_id=element_id,
                    start_time=start_time,
                    end_time=None  # Permanent trip
                )
                
            elif fault_type == 'load_trip':
                # Load trip
                load = self.network.get_load(element_id)
                if not load:
                    print(f"Load {element_id} not found")
                    return False
                
                contingency.add_load_trip(
                    load_id=element_id,
                    start_time=start_time,
                    end_time=None  # Permanent trip
                )
                
            elif fault_type == 'transformer_trip':
                # Transformer trip
                transformer = self.network.get_transformer(element_id)
                if not transformer:
                    print(f"Transformer {element_id} not found")
                    return False
                
                contingency.add_transformer_trip(
                    transformer_id=element_id,
                    start_time=start_time,
                    end_time=None  # Permanent trip
                )
                
            else:
                print(f"Unsupported fault type: {fault_type}")
                return False
            
            # Add the contingency to the simulation
            self.simulation.add_contingency(contingency)
            print(f"Successfully applied {fault_type} to {element_id}")
            return True
            
        except Exception as e:
            print(f"Error applying fault: {str(e)}")
            return False
    
    def run_simulation(self, duration=10.0, time_step=0.01):
        """
        Run a time-domain simulation for transient stability assessment.
        
        Args:
            duration: Simulation duration (seconds)
            time_step: Simulation time step (seconds)
            
        Returns:
            Dictionary containing simulation results
        """
        print(f"Running transient stability simulation ({duration}s)...")
        
        # Set simulation parameters
        self.simulation.set_duration(duration)
        self.simulation.set_time_step(time_step)
        
        # Run the simulation
        result = self.simulation.run()
        
        if result:
            print("Transient stability simulation completed successfully")
            
            # Extract generator rotor angles
            generators = self.network.get_generators()
            gen_angles = {}
            gen_speeds = {}
            
            for gen in generators:
                gen_id = gen.get_id()
                if hasattr(gen, 'get_rotor_angle'):
                    angles = np.array(gen.get_variable_values('rotor_angle'))
                    gen_angles[gen_id] = angles
                
                if hasattr(gen, 'get_rotor_speed'):
                    speeds = np.array(gen.get_variable_values('rotor_speed'))
                    gen_speeds[gen_id] = speeds
            
            # Extract bus voltages
            buses = self.network.get_buses()
            bus_voltages = {}
            
            for bus in buses:
                bus_id = bus.get_id()
                if bus.is_connected():
                    voltages = np.array(bus.get_variable_values('v'))
                    v_nom = bus.get_v_nom()
                    voltages_pu = voltages / v_nom
                    
                    bus_voltages[bus_id] = {
                        'voltage': voltages,
                        'voltage_pu': voltages_pu
                    }
            
            # Store time array
            time = np.linspace(0, duration, len(next(iter(gen_angles.values()))) if gen_angles else 1)
            
            self.results = {
                'success': True,
                'time': time,
                'generator_angles': gen_angles,
                'generator_speeds': gen_speeds,
                'bus_voltages': bus_voltages
            }
            
            return self.results
        else:
            print("Transient stability simulation failed")
            self.results = {'success': False}
            return self.results
    
    def calculate_transient_stability_index(self):
        """
        Calculate the Transient Stability Index (TSI).
        
        TSI = ((360 - delta_max) / (360 + delta_max)) * 100%
        
        Where delta_max is the maximum angular separation between any two rotor angles in degrees.
        
        Returns:
            TSI value (percentage)
        """
        if self.results is None or not self.results['success']:
            print("No valid simulation results available")
            return None
        
        gen_angles = self.results['generator_angles']
        
        if not gen_angles:
            print("No generator angle data available")
            return None
        
        # Calculate maximum angle separation for each time step
        angle_separations = []
        for t in range(len(self.results['time'])):
            angles_at_t = [angles[t] for angles in gen_angles.values()]
            if angles_at_t:
                max_angle = max(angles_at_t)
                min_angle = min(angles_at_t)
                separation = max_angle - min_angle
                angle_separations.append(separation)
        
        # Find the maximum separation across all time steps
        if angle_separations:
            max_separation = max(angle_separations)
            
            # Calculate TSI
            tsi = ((360 - max_separation) / (360 + max_separation)) * 100
            
            print(f"Maximum angle separation: {max_separation:.2f} degrees")
            print(f"Transient Stability Index (TSI): {tsi:.2f}%")
            
            return tsi
        else:
            print("Could not calculate angle separations")
            return None
    
    def check_generator_synchronism(self, threshold_degrees=180):
        """
        Check if generators maintain synchronism.
        
        Args:
            threshold_degrees: Angle threshold in degrees to consider loss of synchronism
            
        Returns:
            Dictionary containing synchronism assessment
        """
        if self.results is None or not self.results['success']:
            print("No valid simulation results available")
            return {'stable': False, 'message': 'No valid simulation results'}
        
        gen_angles = self.results['generator_angles']
        
        if not gen_angles:
            print("No generator angle data available")
            return {'stable': False, 'message': 'No generator angle data'}
        
        # Check angle differences
        max_separation = 0
        unstable_gens = []
        
        for t in range(len(self.results['time'])):
            angles_at_t = {}
            for gen_id, angles in gen_angles.items():
                angles_at_t[gen_id] = angles[t]
            
            for gen1_id, angle1 in angles_at_t.items():
                for gen2_id, angle2 in angles_at_t.items():
                    if gen1_id != gen2_id:
                        separation = abs(angle1 - angle2)
                        if separation > max_separation:
                            max_separation = separation
                        
                        if separation > threshold_degrees:
                            if gen1_id not in unstable_gens:
                                unstable_gens.append(gen1_id)
                            if gen2_id not in unstable_gens:
                                unstable_gens.append(gen2_id)
        
        is_stable = len(unstable_gens) == 0
        
        if is_stable:
            message = f"All generators maintain synchronism (max separation: {max_separation:.2f} degrees)"
        else:
            message = f"{len(unstable_gens)} generators lose synchronism (max separation: {max_separation:.2f} degrees)"
        
        return {
            'stable': is_stable,
            'max_separation': max_separation,
            'unstable_generators': unstable_gens,
            'message': message
        }
    
    def calculate_critical_clearing_time(self, fault_type, element_id, 
                                        start_time=1.0, 
                                        initial_duration=0.1,
                                        max_duration=1.0,
                                        step=0.05,
                                        tsi_threshold=10,
                                        **fault_params):
        """
        Calculate the Critical Clearing Time (CCT) for a given fault.
        
        Args:
            fault_type: Type of fault
            element_id: ID of the element to apply fault to
            start_time: Fault start time
            initial_duration: Initial fault duration to try
            max_duration: Maximum fault duration to try
            step: Step size for increasing fault duration
            tsi_threshold: TSI threshold to consider unstable (%)
            **fault_params: Additional fault parameters
            
        Returns:
            Critical clearing time (seconds)
        """
        print(f"Calculating critical clearing time for {fault_type} on {element_id}...")
        
        # Create a copy of the simulation to avoid modifying the original
        sim_copy = self.simulation.copy()
        analyzer = TransientStabilityAnalyzer(sim_copy)
        
        duration = initial_duration
        last_stable_duration = None
        first_unstable_duration = None
        
        while duration <= max_duration:
            # Apply fault
            analyzer.apply_fault(fault_type, element_id, start_time, duration, **fault_params)
            
            # Run simulation
            analyzer.run_simulation()
            
            # Calculate TSI
            tsi = analyzer.calculate_transient_stability_index()
            
            if tsi is None or tsi < tsi_threshold:
                print(f"Duration {duration}s: Unstable (TSI: {tsi}%)")
                first_unstable_duration = duration
                break
            else:
                print(f"Duration {duration}s: Stable (TSI: {tsi}%)")
                last_stable_duration = duration
            
            # Increase duration
            duration += step
            
            # Reset simulation for next iteration
            sim_copy = self.simulation.copy()
            analyzer = TransientStabilityAnalyzer(sim_copy)
        
        # If all durations were stable
        if first_unstable_duration is None:
            print(f"System remains stable up to {max_duration}s")
            return max_duration
        
        # If the initial duration was unstable
        if last_stable_duration is None:
            print(f"System is unstable at initial duration {initial_duration}s")
            return 0
        
        # Refine CCT using bisection method
        lower = last_stable_duration
        upper = first_unstable_duration
        
        print(f"Refining CCT between {lower}s and {upper}s")
        
        while upper - lower > 0.01:  # 10ms precision
            mid = (lower + upper) / 2
            
            # Apply fault with mid duration
            sim_copy = self.simulation.copy()
            analyzer = TransientStabilityAnalyzer(sim_copy)
            analyzer.apply_fault(fault_type, element_id, start_time, mid, **fault_params)
            
            # Run simulation
            analyzer.run_simulation()
            
            # Calculate TSI
            tsi = analyzer.calculate_transient_stability_index()
            
            if tsi is None or tsi < tsi_threshold:
                print(f"Duration {mid}s: Unstable (TSI: {tsi}%)")
                upper = mid
            else:
                print(f"Duration {mid}s: Stable (TSI: {tsi}%)")
                lower = mid
        
        cct = lower
        print(f"Critical Clearing Time: {cct:.3f}s")
        return cct
    
    def assess_transient_stability(self, tsi_threshold=10):
        """
        Assess transient stability based on simulation results.
        
        Args:
            tsi_threshold: TSI threshold to consider stable (%)
            
        Returns:
            Dictionary containing assessment results
        """
        print("Assessing transient stability...")
        
        if self.results is None or not self.results['success']:
            return {'stable': False, 'message': 'No valid simulation results'}
        
        # Calculate TSI
        tsi = self.calculate_transient_stability_index()
        
        # Check synchronism
        sync_check = self.check_generator_synchronism()
        
        # Determine stability
        is_stable = (tsi is not None and tsi >= tsi_threshold and sync_check['stable'])
        
        if is_stable:
            message = f"System is transiently stable (TSI: {tsi:.2f}%)"
        else:
            reasons = []
            if tsi is None:
                reasons.append("could not calculate TSI")
            elif tsi < tsi_threshold:
                reasons.append(f"TSI ({tsi:.2f}%) below threshold ({tsi_threshold}%)")
            
            if not sync_check['stable']:
                reasons.append("loss of synchronism")
            
            message = f"System is transiently unstable: {', '.join(reasons)}"
        
        # Compile results
        assessment = {
            'stable': is_stable,
            'tsi': tsi,
            'tsi_threshold': tsi_threshold,
            'synchronism': sync_check,
            'message': message
        }
        
        print(message)
        return assessment


# Example usage
if __name__ == "__main__":
    # This requires a simulation object to be created first
    # from load_ieee_systems import IEEESystemLoader
    # loader = IEEESystemLoader()
    # sim = loader.load_ieee68()
    # analyzer = TransientStabilityAnalyzer(sim)
    # analyzer.apply_fault('line_trip', 'LINE_1')
    # analyzer.run_simulation()
    # results = analyzer.assess_transient_stability()
    pass