"""
Module for injecting different types of faults into power systems for stability analysis.
"""

import logging
import random
from copy import deepcopy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_bus_fault(system, bus_idx, fault_time=1.0, clear_time=1.1, r=0.0, x=0.01):
    """
    Apply a three-phase bus fault to the system.
    
    Args:
        system (andes.System): ANDES system object
        bus_idx (int): Index of the bus where the fault will be applied
        fault_time (float): Time when the fault starts (seconds)
        clear_time (float): Time when the fault is cleared (seconds)
        r (float): Fault resistance (p.u.)
        x (float): Fault reactance (p.u.)
        
    Returns:
        andes.System: System with the bus fault added
    """
    try:
        # Create a deep copy to avoid modifying the original system
        system_copy = deepcopy(system)
        
        # Check if bus exists
        if bus_idx >= system_copy.Bus.n:
            raise ValueError(f"Bus index {bus_idx} does not exist")
        
        # Add bus fault event
        system_copy.TDS.add_event('bus_fault', 
                             {'bus': bus_idx, 'tf': fault_time, 'tc': clear_time, 'r': r, 'x': x})
        
        logger.info(f"Applied three-phase fault on bus {bus_idx} at t={fault_time}s, cleared at t={clear_time}s")
        return system_copy
    
    except Exception as e:
        logger.error(f"Failed to apply bus fault: {str(e)}")
        raise

def apply_line_trip(system, line_idx, trip_time=1.0, reconnect_time=None):
    """
    Apply a line trip event to the system.
    
    Args:
        system (andes.System): ANDES system object
        line_idx (int): Index of the line to trip
        trip_time (float): Time when the line trips (seconds)
        reconnect_time (float, optional): Time when the line reconnects (seconds)
                                          If None, the line stays tripped
        
    Returns:
        andes.System: System with the line trip added
    """
    try:
        # Create a deep copy to avoid modifying the original system
        system_copy = deepcopy(system)
        
        # Check if line exists
        if line_idx >= system_copy.Line.n:
            raise ValueError(f"Line index {line_idx} does not exist")
        
        # Add line trip event
        system_copy.TDS.add_event('line_trip', {'line': line_idx, 'tf': trip_time})
        
        # Add line reconnect event if specified
        if reconnect_time is not None:
            system_copy.TDS.add_event('line_reconnect', {'line': line_idx, 'tf': reconnect_time})
            logger.info(f"Applied line trip for line {line_idx} at t={trip_time}s, reconnected at t={reconnect_time}s")
        else:
            logger.info(f"Applied line trip for line {line_idx} at t={trip_time}s (permanent)")
        
        return system_copy
    
    except Exception as e:
        logger.error(f"Failed to apply line trip: {str(e)}")
        raise

def apply_line_fault(system, line_idx, fault_time=1.0, clear_time=1.1, location=0.5, r=0.0, x=0.01, trip_line=True):
    """
    Apply a line fault followed by optional line trip.
    
    Args:
        system (andes.System): ANDES system object
        line_idx (int): Index of the line where the fault will be applied
        fault_time (float): Time when the fault starts (seconds)
        clear_time (float): Time when the fault is cleared (seconds)
        location (float): Location of the fault (0-1, from the 'from' bus to the 'to' bus)
        r (float): Fault resistance (p.u.)
        x (float): Fault reactance (p.u.)
        trip_line (bool): Whether to trip the line after fault clearing
        
    Returns:
        andes.System: System with the line fault added
    """
    try:
        # Create a deep copy to avoid modifying the original system
        system_copy = deepcopy(system)
        
        # Check if line exists
        if line_idx >= system_copy.Line.n:
            raise ValueError(f"Line index {line_idx} does not exist")
        
        # Add line fault event
        system_copy.TDS.add_event('line_fault', 
                             {'line': line_idx, 'tf': fault_time, 'tc': clear_time, 
                              'loc': location, 'r': r, 'x': x})
        
        # Add line trip event if specified
        if trip_line:
            system_copy.TDS.add_event('line_trip', {'line': line_idx, 'tf': clear_time})
            logger.info(f"Applied fault on line {line_idx} at t={fault_time}s, cleared and tripped at t={clear_time}s")
        else:
            logger.info(f"Applied fault on line {line_idx} at t={fault_time}s, cleared at t={clear_time}s (no trip)")
        
        return system_copy
    
    except Exception as e:
        logger.error(f"Failed to apply line fault: {str(e)}")
        raise

def apply_generator_trip(system, gen_idx, trip_time=1.0):
    """
    Apply a generator trip event to the system.
    
    Args:
        system (andes.System): ANDES system object
        gen_idx (int): Index of the generator to trip
        trip_time (float): Time when the generator trips (seconds)
        
    Returns:
        andes.System: System with the generator trip added
    """
    try:
        # Create a deep copy to avoid modifying the original system
        system_copy = deepcopy(system)
        
        # Check if the GENROU model is being used
        if not hasattr(system_copy, 'GENROU'):
            raise ValueError("GENROU generator model not found in the system")
        
        # Check if generator exists
        if gen_idx >= system_copy.GENROU.n:
            raise ValueError(f"Generator index {gen_idx} does not exist")
        
        # Add generator trip event
        system_copy.TDS.add_event('gen_trip', {'gen': gen_idx, 'tf': trip_time})
        
        logger.info(f"Applied generator trip for generator {gen_idx} at t={trip_time}s")
        return system_copy
    
    except Exception as e:
        logger.error(f"Failed to apply generator trip: {str(e)}")
        raise

def apply_load_change(system, load_idx, change_time=1.0, scale_p=0.5, scale_q=0.5):
    """
    Apply a load change event to the system.
    
    Args:
        system (andes.System): ANDES system object
        load_idx (int): Index of the load to change
        change_time (float): Time when the load changes (seconds)
        scale_p (float): Scaling factor for active power (1.0 = no change)
        scale_q (float): Scaling factor for reactive power (1.0 = no change)
        
    Returns:
        andes.System: System with the load change added
    """
    try:
        # Create a deep copy to avoid modifying the original system
        system_copy = deepcopy(system)
        
        # Check if the PQ model is being used for loads
        if not hasattr(system_copy, 'PQ'):
            raise ValueError("PQ load model not found in the system")
        
        # Check if load exists
        if load_idx >= system_copy.PQ.n:
            raise ValueError(f"Load index {load_idx} does not exist")
        
        # Add load change event
        system_copy.TDS.add_event('alter', {'model': 'PQ', 'idx': load_idx, 
                                        'par': 'p0', 'tf': change_time, 
                                        'val': system_copy.PQ.p0[load_idx] * scale_p})
        
        system_copy.TDS.add_event('alter', {'model': 'PQ', 'idx': load_idx, 
                                        'par': 'q0', 'tf': change_time, 
                                        'val': system_copy.PQ.q0[load_idx] * scale_q})
        
        logger.info(f"Applied load change for load {load_idx} at t={change_time}s (P scale: {scale_p}, Q scale: {scale_q})")
        return system_copy
    
    except Exception as e:
        logger.error(f"Failed to apply load change: {str(e)}")
        raise

def apply_random_fault(system, fault_type=None, fault_time=1.0, clear_time=1.1):
    """
    Apply a random fault to the system.
    
    Args:
        system (andes.System): ANDES system object
        fault_type (str, optional): Type of fault ('bus', 'line', 'generator', 'load')
                                    If None, a random type will be selected
        fault_time (float): Time when the fault starts (seconds)
        clear_time (float): Time when the fault is cleared (seconds)
        
    Returns:
        andes.System: System with a random fault added
        dict: Information about the applied fault
    """
    try:
        # Create a deep copy to avoid modifying the original system
        system_copy = deepcopy(system)
        
        # Determine fault type if not specified
        fault_types = ['bus', 'line', 'generator', 'load']
        if fault_type is None or fault_type not in fault_types:
            fault_type = random.choice(fault_types)
        
        fault_info = {'type': fault_type, 'time': fault_time, 'clear_time': clear_time}
        
        # Apply the selected fault type
        if fault_type == 'bus':
            # Apply bus fault
            bus_idx = random.randint(0, system_copy.Bus.n - 1)
            system_copy = apply_bus_fault(system_copy, bus_idx, fault_time, clear_time)
            fault_info['bus_idx'] = bus_idx
            
        elif fault_type == 'line':
            # Apply line fault
            line_idx = random.randint(0, system_copy.Line.n - 1)
            trip_line = random.choice([True, False])
            system_copy = apply_line_fault(system_copy, line_idx, fault_time, clear_time, trip_line=trip_line)
            fault_info['line_idx'] = line_idx
            fault_info['trip_line'] = trip_line
            
        elif fault_type == 'generator' and hasattr(system_copy, 'GENROU'):
            # Apply generator trip
            gen_idx = random.randint(0, system_copy.GENROU.n - 1)
            system_copy = apply_generator_trip(system_copy, gen_idx, fault_time)
            fault_info['gen_idx'] = gen_idx
            
        elif fault_type == 'load' and hasattr(system_copy, 'PQ'):
            # Apply load change
            load_idx = random.randint(0, system_copy.PQ.n - 1)
            scale_p = random.uniform(0.2, 1.8)  # Random load change between 20% and 180%
            scale_q = random.uniform(0.2, 1.8)
            system_copy = apply_load_change(system_copy, load_idx, fault_time, scale_p, scale_q)
            fault_info['load_idx'] = load_idx
            fault_info['scale_p'] = scale_p
            fault_info['scale_q'] = scale_q
        
        logger.info(f"Applied random {fault_type} fault")
        return system_copy, fault_info
    
    except Exception as e:
        logger.error(f"Failed to apply random fault: {str(e)}")
        raise

def generate_fault_scenarios(system, n_scenarios=10, fault_types=None):
    """
    Generate multiple fault scenarios for comprehensive stability assessment.
    
    Args:
        system (andes.System): ANDES system object
        n_scenarios (int): Number of fault scenarios to generate
        fault_types (list, optional): List of fault types to include
                                       If None, all types will be included
        
    Returns:
        list: List of (system, fault_info) tuples with fault scenarios
    """
    try:
        scenarios = []
        
        # Determine fault types to include
        all_fault_types = ['bus', 'line', 'generator', 'load']
        if fault_types is None:
            fault_types = all_fault_types
        else:
            # Ensure all specified fault types are valid
            for ft in fault_types:
                if ft not in all_fault_types:
                    logger.warning(f"Invalid fault type '{ft}' specified, ignoring")
                    fault_types.remove(ft)
        
        # Generate the requested number of scenarios
        for i in range(n_scenarios):
            # Select a random fault type from the specified types
            fault_type = random.choice(fault_types)
            
            # Generate random fault timing
            fault_time = random.uniform(0.5, 2.0)  # Between 0.5s and 2.0s
            clear_time = fault_time + random.uniform(0.05, 0.3)  # Fault duration between 0.05s and 0.3s
            
            # Apply the random fault
            faulted_system, fault_info = apply_random_fault(system, fault_type, fault_time, clear_time)
            
            # Add to scenarios
            scenarios.append((faulted_system, fault_info))
            
            logger.info(f"Generated scenario {i+1}/{n_scenarios}: {fault_type} fault")
        
        return scenarios
    
    except Exception as e:
        logger.error(f"Failed to generate fault scenarios: {str(e)}")
        raise

def apply_multiple_faults(system, fault_sequence):
    """
    Apply a sequence of multiple faults to the system.
    
    Args:
        system (andes.System): ANDES system object
        fault_sequence (list): List of fault dictionaries with the following format:
                               [
                                   {
                                       'type': 'bus',
                                       'bus_idx': 1,
                                       'fault_time': 1.0,
                                       'clear_time': 1.1
                                   },
                                   {
                                       'type': 'line',
                                       'line_idx': 2,
                                       'fault_time': 3.0,
                                       'clear_time': 3.1,
                                       'trip_line': True
                                   },
                                   ...
                               ]
        
    Returns:
        andes.System: System with multiple faults added
    """
    try:
        # Create a deep copy to avoid modifying the original system
        system_copy = deepcopy(system)
        
        # Apply each fault in the sequence
        for i, fault in enumerate(fault_sequence):
            fault_type = fault.get('type')
            fault_time = fault.get('fault_time', 1.0)
            clear_time = fault.get('clear_time', fault_time + 0.1)
            
            if fault_type == 'bus':
                bus_idx = fault.get('bus_idx')
                if bus_idx is None:
                    raise ValueError(f"Bus index not specified for bus fault at position {i}")
                r = fault.get('r', 0.0)
                x = fault.get('x', 0.01)
                system_copy = apply_bus_fault(system_copy, bus_idx, fault_time, clear_time, r, x)
                
            elif fault_type == 'line':
                line_idx = fault.get('line_idx')
                if line_idx is None:
                    raise ValueError(f"Line index not specified for line fault at position {i}")
                    
                if 'trip' in fault:
                    # This is a simple line trip without fault
                    trip_time = fault.get('fault_time')
                    reconnect_time = fault.get('clear_time') if not fault.get('permanent', False) else None
                    system_copy = apply_line_trip(system_copy, line_idx, trip_time, reconnect_time)
                else:
                    # This is a line fault
                    location = fault.get('location', 0.5)
                    r = fault.get('r', 0.0)
                    x = fault.get('x', 0.01)
                    trip_line = fault.get('trip_line', True)
                    system_copy = apply_line_fault(system_copy, line_idx, fault_time, clear_time, location, r, x, trip_line)
                
            elif fault_type == 'generator':
                gen_idx = fault.get('gen_idx')
                if gen_idx is None:
                    raise ValueError(f"Generator index not specified for generator fault at position {i}")
                system_copy = apply_generator_trip(system_copy, gen_idx, fault_time)
                
            elif fault_type == 'load':
                load_idx = fault.get('load_idx')
                if load_idx is None:
                    raise ValueError(f"Load index not specified for load fault at position {i}")
                scale_p = fault.get('scale_p', 0.5)
                scale_q = fault.get('scale_q', 0.5)
                system_copy = apply_load_change(system_copy, load_idx, fault_time, scale_p, scale_q)
                
            else:
                logger.warning(f"Unknown fault type '{fault_type}' at position {i}, skipping")
        
        logger.info(f"Applied {len(fault_sequence)} faults to the system")
        return system_copy
    
    except Exception as e:
        logger.error(f"Failed to apply multiple faults: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    import load_ieee_system
    
    # Load IEEE 68-bus system
    system = load_ieee_system.load_ieee68()
    
    # Example 1: Apply a bus fault
    if system.Bus.n > 0:
        faulted_system = apply_bus_fault(system, 0, fault_time=1.0, clear_time=1.1)
        print(f"Applied bus fault to bus 0")
    
    # Example 2: Apply a line trip
    if system.Line.n > 0:
        faulted_system = apply_line_trip(system, 0, trip_time=1.0)
        print(f"Applied line trip to line 0")
    
    # Example 3: Generate multiple random fault scenarios
    scenarios = generate_fault_scenarios(system, n_scenarios=3)
    for i, (faulted_sys, fault_info) in enumerate(scenarios):
        print(f"\nScenario {i+1}:")
        for key, value in fault_info.items():
            print(f"  {key}: {value}")
    
    # Example 4: Apply a sequence of faults
    fault_sequence = [
        {'type': 'bus', 'bus_idx': 0, 'fault_time': 1.0, 'clear_time': 1.1},
        {'type': 'line', 'line_idx': 0, 'fault_time': 2.0, 'clear_time': 2.1, 'trip_line': True}
    ]
    faulted_system = apply_multiple_faults(system, fault_sequence)
    print("\nApplied a sequence of multiple faults")