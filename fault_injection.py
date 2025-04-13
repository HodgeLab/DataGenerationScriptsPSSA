"""
Module for injecting different types of faults into Dynawo power system models.
"""

import dynawo
import numpy as np
import random
from enum import Enum


class FaultType(Enum):
    """Enumeration of different fault types."""
    BUS_FAULT = "bus_fault"
    LINE_TRIP = "line_trip"
    LINE_FAULT = "line_fault"
    GENERATOR_TRIP = "generator_trip"
    LOAD_TRIP = "load_trip"
    TRANSFORMER_TRIP = "transformer_trip"
    BREAKER_FAILURE = "breaker_failure"
    MULTIPLE_LINE_TRIP = "multiple_line_trip"
    CASCADING_FAILURE = "cascading_failure"


class FaultInjector:
    """Class to inject various types of faults into Dynawo power system models."""
    
    def __init__(self, simulation):
        """
        Initialize the fault injector.
        
        Args:
            simulation: A Dynawo simulation object
        """
        self.simulation = simulation
        self.network = simulation.get_network()
        self.applied_faults = []
    
    def clear_all_faults(self):
        """
        Clear all previously applied faults.
        """
        print("Clearing all faults...")
        self.simulation.clear_contingencies()
        self.applied_faults = []
        print("All faults cleared")
    
    def apply_bus_fault(self, bus_id=None, start_time=1.0, duration=0.1, resistance=0.0):
        """
        Apply a three-phase short circuit fault to a bus.
        
        Args:
            bus_id: ID of the bus (if None, select a random bus)
            start_time: Fault start time (seconds)
            duration: Fault duration (seconds)
            resistance: Fault resistance (ohms)
            
        Returns:
            Dictionary containing fault details
        """
        if bus_id is None:
            # Select a random bus
            buses = list(self.network.get_buses())
            bus = random.choice(buses)
            bus_id = bus.get_id()
        
        print(f"Applying bus fault to {bus_id} at t={start_time}s for {duration}s")
        
        # Create a contingency
        contingency = dynawo.Contingency(self.simulation)
        
        # Add bus fault
        contingency.add_bus_fault(
            bus_id=bus_id,
            start_time=start_time,
            end_time=start_time + duration,
            resistance=resistance
        )
        
        # Add to simulation
        self.simulation.add_contingency(contingency)
        
        # Record fault
        fault_details = {
            'type': FaultType.BUS_FAULT.value,
            'element_id': bus_id,
            'start_time': start_time,
            'duration': duration,
            'parameters': {'resistance': resistance}
        }
        self.applied_faults.append(fault_details)
        
        return fault_details
    
    def apply_line_trip(self, line_id=None, start_time=1.0, permanent=True, reconnect_time=None):
        """
        Trip a transmission line.
        
        Args:
            line_id: ID of the line (if None, select a random line)
            start_time: Fault start time (seconds)
            permanent: Whether the trip is permanent
            reconnect_time: Time to reconnect the line (only if not permanent)
            
        Returns:
            Dictionary containing fault details
        """
        if line_id is None:
            # Select a random line
            lines = list(self.network.get_lines())
            line = random.choice(lines)
            line_id = line.get_id()
        
        print(f"Applying line trip to {line_id} at t={start_time}s")
        
        # Create a contingency
        contingency = dynawo.Contingency(self.simulation)
        
        # Set end time
        end_time = None
        if not permanent and reconnect_time is not None:
            end_time = reconnect_time
        
        # Add line trip
        contingency.add_line_trip(
            line_id=line_id,
            start_time=start_time,
            end_time=end_time
        )
        
        # Add to simulation
        self.simulation.add_contingency(contingency)
        
        # Record fault
        fault_details = {
            'type': FaultType.LINE_TRIP.value,
            'element_id': line_id,
            'start_time': start_time,
            'permanent': permanent,
            'parameters': {'reconnect_time': reconnect_time}
        }
        self.applied_faults.append(fault_details)
        
        return fault_details
    
    def apply_line_fault(self, line_id=None, start_time=1.0, duration=0.1, location=50.0, 
                        resistance=0.0, trip_line=True, trip_time=None):
        """
        Apply a fault on a transmission line.
        
        Args:
            line_id: ID of the line (if None, select a random line)
            start_time: Fault start time (seconds)
            duration: Fault duration (seconds)
            location: Location along the line (% from bus1)
            resistance: Fault resistance (ohms)
            trip_line: Whether to trip the line after the fault
            trip_time: Time to trip the line (if None, use start_time + duration)
            
        Returns:
            Dictionary containing fault details
        """
        if line_id is None:
            # Select a random line
            lines = list(self.network.get_lines())
            line = random.choice(lines)
            line_id = line.get_id()
        
        print(f"Applying line fault to {line_id} at t={start_time}s for {duration}s")
        
        # Create a contingency
        contingency = dynawo.Contingency(self.simulation)
        
        # Add line fault
        contingency.add_line_fault(
            line_id=line_id,
            start_time=start_time,
            end_time=start_time + duration,
            location=location,
            resistance=resistance
        )
        
        # Trip the line if required
        if trip_line:
            if trip_time is None:
                trip_time = start_time + duration
                
            contingency.add_line_trip(
                line_id=line_id,
                start_time=trip_time,
                end_time=None  # Permanent trip
            )
        
        # Add to simulation
        self.simulation.add_contingency(contingency)
        
        # Record fault
        fault_details = {
            'type': FaultType.LINE_FAULT.value,
            'element_id': line_id,
            'start_time': start_time,
            'duration': duration,
            'parameters': {
                'location': location,
                'resistance': resistance,
                'trip_line': trip_line,
                'trip_time': trip_time
            }
        }
        self.applied_faults.append(fault_details)
        
        return fault_details
    
    def apply_generator_trip(self, generator_id=None, start_time=1.0, permanent=True):
        """
        Trip a generator.
        
        Args:
            generator_id: ID of the generator (if None, select a random generator)
            start_time: Fault start time (seconds)
            permanent: Whether the trip is permanent
            
        Returns:
            Dictionary containing fault details
        """
        if generator_id is None:
            # Select a random generator
            generators = list(self.network.get_generators())
            generator = random.choice(generators)
            generator_id = generator.get_id()
        
        print(f"Applying generator trip to {generator_id} at t={start_time}s")
        
        # Create a contingency
        contingency = dynawo.Contingency(self.simulation)
        
        # Add generator trip
        contingency.add_generator_trip(
            generator_id=generator_id,
            start_time=start_time,
            end_time=None if permanent else start_time + 10.0  # Arbitrary reconnection time
        )
        
        # Add to simulation
        self.simulation.add_contingency(contingency)
        
        # Record fault
        fault_details = {
            'type': FaultType.GENERATOR_TRIP.value,
            'element_id': generator_id,
            'start_time': start_time,
            'permanent': permanent
        }
        self.applied_faults.append(fault_details)
        
        return fault_details
    
    def apply_load_trip(self, load_id=None, start_time=1.0, permanent=True, percentage=100.0):
        """
        Trip a load.
        
        Args:
            load_id: ID of the load (if None, select a random load)
            start_time: Fault start time (seconds)
            permanent: Whether the trip is permanent
            percentage: Percentage of load to trip (0-100)
            
        Returns:
            Dictionary containing fault details
        """
        if load_id is None:
            # Select a random load
            loads = list(self.network.get_loads())
            load = random.choice(loads)
            load_id = load.get_id()
        
        print(f"Applying load trip to {load_id} at t={start_time}s")
        
        # Create a contingency
        contingency = dynawo.Contingency(self.simulation)
        
        # Add load trip
        contingency.add_load_trip(
            load_id=load_id,
            start_time=start_time,
            end_time=None if permanent else start_time + 10.0,  # Arbitrary reconnection time
            percentage=percentage
        )
        
        # Add to simulation
        self.simulation.add_contingency(contingency)
        
        # Record fault
        fault_details = {
            'type': FaultType.LOAD_TRIP.value,
            'element_id': load_id,
            'start_time': start_time,
            'permanent': permanent,
            'parameters': {'percentage': percentage}
        }
        self.applied_faults.append(fault_details)
        
        return fault_details