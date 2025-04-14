"""
Module for performing topological changes to power systems in ANDES.
"""

import logging
import numpy as np
import pandas as pd
from copy import deepcopy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def modify_line_capacity(system, line_idx, new_capacity_mva):
    """
    Modify the capacity of a transmission line.
    
    Args:
        system (andes.System): ANDES system object
        line_idx (int or str): Line index or name
        new_capacity_mva (float): New line capacity in MVA
        
    Returns:
        andes.System: Modified ANDES system object
    """
    try:
        # Create a deep copy to avoid modifying the original system
        modified_system = deepcopy(system)
        
        # Convert line_idx to int if it's a string identifier
        if isinstance(line_idx, str):
            line_idx = system.Line.idx2uid(line_idx)
        
        # Update the line rating
        modified_system.Line.rate_a[line_idx] = new_capacity_mva
        
        logger.info(f"Modified line {line_idx} capacity to {new_capacity_mva} MVA")
        return modified_system
    
    except Exception as e:
        logger.error(f"Failed to modify line capacity: {str(e)}")
        raise

def disconnect_line(system, line_idx):
    """
    Disconnect a transmission line from the system.
    
    Args:
        system (andes.System): ANDES system object
        line_idx (int or str): Line index or name
        
    Returns:
        andes.System: Modified ANDES system object
    """
    try:
        # Create a deep copy to avoid modifying the original system
        modified_system = deepcopy(system)
        
        # Convert line_idx to int if it's a string identifier
        if isinstance(line_idx, str):
            line_idx = system.Line.idx2uid(line_idx)
        
        # Set line status to 0 (disconnected)
        modified_system.Line.u[line_idx] = 0
        
        logger.info(f"Disconnected line {line_idx}")
        return modified_system
    
    except Exception as e:
        logger.error(f"Failed to disconnect line: {str(e)}")
        raise

def connect_line(system, line_idx):
    """
    Connect a previously disconnected transmission line.
    
    Args:
        system (andes.System): ANDES system object
        line_idx (int or str): Line index or name
        
    Returns:
        andes.System: Modified ANDES system object
    """
    try:
        # Create a deep copy to avoid modifying the original system
        modified_system = deepcopy(system)
        
        # Convert line_idx to int if it's a string identifier
        if isinstance(line_idx, str):
            line_idx = system.Line.idx2uid(line_idx)
        
        # Set line status to 1 (connected)
        modified_system.Line.u[line_idx] = 1
        
        logger.info(f"Connected line {line_idx}")
        return modified_system
    
    except Exception as e:
        logger.error(f"Failed to connect line: {str(e)}")
        raise

def modify_line_impedance(system, line_idx, new_r=None, new_x=None, new_b=None):
    """
    Modify the impedance parameters of a transmission line.
    
    Args:
        system (andes.System): ANDES system object
        line_idx (int or str): Line index or name
        new_r (float, optional): New resistance value in p.u.
        new_x (float, optional): New reactance value in p.u.
        new_b (float, optional): New susceptance value in p.u.
        
    Returns:
        andes.System: Modified ANDES system object
    """
    try:
        # Create a deep copy to avoid modifying the original system
        modified_system = deepcopy(system)
        
        # Convert line_idx to int if it's a string identifier
        if isinstance(line_idx, str):
            line_idx = system.Line.idx2uid(line_idx)
        
        # Update impedance parameters if provided
        if new_r is not None:
            modified_system.Line.r[line_idx] = new_r
        
        if new_x is not None:
            modified_system.Line.x[line_idx] = new_x
        
        if new_b is not None:
            modified_system.Line.b[line_idx] = new_b
        
        logger.info(f"Modified line {line_idx} impedance parameters")
        return modified_system
    
    except Exception as e:
        logger.error(f"Failed to modify line impedance: {str(e)}")
        raise

def disconnect_bus(system, bus_idx):
    """
    Disconnect a bus by disconnecting all its connected elements.
    
    Args:
        system (andes.System): ANDES system object
        bus_idx (int or str): Bus index or name
        
    Returns:
        andes.System: Modified ANDES system object
    """
    try:
        # Create a deep copy to avoid modifying the original system
        modified_system = deepcopy(system)
        
        # Convert bus_idx to int if it's a string identifier
        if isinstance(bus_idx, str):
            bus_idx = system.Bus.idx2uid(bus_idx)
        
        # Get the lines connected to this bus
        connected_lines = []
        for i in range(modified_system.Line.n):
            if modified_system.Line.bus1[i] == bus_idx or modified_system.Line.bus2[i] == bus_idx:
                connected_lines.append(i)
        
        # Disconnect all lines connected to this bus
        for line_idx in connected_lines:
            modified_system.Line.u[line_idx] = 0
        
        # Set all generators at this bus to 0
        if hasattr(modified_system, 'GENROU'):
            for i in range(modified_system.GENROU.n):
                if modified_system.GENROU.bus[i] == bus_idx:
                    modified_system.GENROU.u[i] = 0
        
        # Set all loads at this bus to 0
        if hasattr(modified_system, 'PQ'):
            for i in range(modified_system.PQ.n):
                if modified_system.PQ.bus[i] == bus_idx:
                    modified_system.PQ.u[i] = 0
        
        logger.info(f"Disconnected bus {bus_idx} and all connected elements")
        return modified_system
    
    except Exception as e:
        logger.error(f"Failed to disconnect bus: {str(e)}")
        raise

def add_shunt_compensation(system, bus_idx, mvar):
    """
    Add shunt compensation to a bus.
    
    Args:
        system (andes.System): ANDES system object
        bus_idx (int or str): Bus index or name
        mvar (float): Reactive power compensation in MVAR (positive for capacitive)
        
    Returns:
        andes.System: Modified ANDES system object
    """
    try:
        # Create a deep copy to avoid modifying the original system
        modified_system = deepcopy(system)
        
        # Convert bus_idx to int if it's a string identifier
        if isinstance(bus_idx, str):
            bus_idx = system.Bus.idx2uid(bus_idx)
        
        # Check if there's an existing shunt at this bus
        existing_shunt_idx = None
        if hasattr(modified_system, 'Shunt'):
            for i in range(modified_system.Shunt.n):
                if modified_system.Shunt.bus[i] == bus_idx:
                    existing_shunt_idx = i
                    break
        
        # Modify existing shunt or add a new one
        if existing_shunt_idx is not None:
            modified_system.Shunt.g[existing_shunt_idx] = 0  # Assuming only reactive compensation
            modified_system.Shunt.b[existing_shunt_idx] = mvar / 100  # Convert to p.u.
            logger.info(f"Modified existing shunt at bus {bus_idx} to {mvar} MVAR")
        else:
            # Add a new shunt device - this requires adding a new element
            # Note: This is a simplified approach and might need adjustment based on ANDES version
            if hasattr(modified_system, 'Shunt'):
                modified_system.Shunt.add(idx=f"SHUNT_{bus_idx}", name=f"SHUNT_{bus_idx}", 
                                         bus=bus_idx, g=0, b=mvar/100)
                logger.info(f"Added new shunt compensation at bus {bus_idx} with {mvar} MVAR")
            else:
                logger.warning("Shunt model not available in the system")
        
        return modified_system
    
    except Exception as e:
        logger.error(f"Failed to add shunt compensation: {str(e)}")
        raise

def get_topology_changes_list(system):
    """
    Generate a list of possible topology changes for the system.
    
    Args:
        system (andes.System): ANDES system object
        
    Returns:
        pd.DataFrame: DataFrame with possible topology changes
    """
    try:
        topology_changes = []
        
        # Add line disconnections
        for i in range(system.Line.n):
            line_name = system.Line.name[i] if system.Line.name[i] else f"Line_{i}"
            from_bus = system.Line.bus1[i]
            to_bus = system.Line.bus2[i]
            
            topology_changes.append({
                'change_type': 'disconnect_line',
                'element_idx': i,
                'element_name': line_name,
                'description': f"Disconnect line from bus {from_bus} to bus {to_bus}"
            })
        
        # Add line capacity modifications
        for i in range(system.Line.n):
            line_name = system.Line.name[i] if system.Line.name[i] else f"Line_{i}"
            from_bus = system.Line.bus1[i]
            to_bus = system.Line.bus2[i]
            current_capacity = system.Line.rate_a[i]
            
            topology_changes.append({
                'change_type': 'modify_line_capacity',
                'element_idx': i,
                'element_name': line_name,
                'description': f"Modify capacity of line from bus {from_bus} to bus {to_bus}",
                'current_value': current_capacity,
                'new_value': current_capacity * 1.5  # Example: increase by 50%
            })
        
        return pd.DataFrame(topology_changes)
    
    except Exception as e:
        logger.error(f"Failed to generate topology changes list: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage (requires imported system)
    import load_ieee_system
    
    # Load IEEE 68-bus system
    system = load_ieee_system.load_ieee68()
    
    # Get possible topology changes
    topology_df = get_topology_changes_list(system)
    print(f"Generated {len(topology_df)} possible topology changes")
    print(topology_df.head())
    
    # Example: Disconnect a line
    if system.Line.n > 0:
        modified_system = disconnect_line(system, 0)
        
        # Get system info after modification
        mod_info = load_ieee_system.get_system_info(modified_system)
        print("\nSystem after disconnecting line 0:")
        for key, value in mod_info.items():
            print(f"  {key}: {value}")