"""
Module for loading IEEE 68-bus and IEEE 300-bus test systems in ANDES.
"""

import os
import andes
import logging
import numpy as np
from andes.utils.paths import get_case

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ieee68():
    """
    Load the IEEE 68-bus (New England/New York) system.
    
    Returns:
        andes.System: ANDES system object with the loaded IEEE 68-bus system
    """
    try:
        # The IEEE 68-bus is also known as the NPCC system in ANDES
        case_path = get_case('npcc/npcc.xlsx')
        logger.info(f"Loading IEEE 68-bus system from {case_path}")
        
        # Initialize ANDES system
        system = andes.System()
        system.config.default_config()
        
        # Load the case
        system.load_file(case_path)
        logger.info(f"Successfully loaded IEEE 68-bus system with {system.Bus.n} buses")
        
        return system
    
    except Exception as e:
        logger.error(f"Failed to load IEEE 68-bus system: {str(e)}")
        raise

def load_ieee300():
    """
    Load the IEEE 300-bus test system.
    
    Returns:
        andes.System: ANDES system object with the loaded IEEE 300-bus system
    """
    try:
        # Get the IEEE 300-bus system path
        case_path = get_case('ieee/ieee300.raw')
        logger.info(f"Loading IEEE 300-bus system from {case_path}")
        
        # Initialize ANDES system
        system = andes.System()
        system.config.default_config()
        
        # Load the case
        system.load_file(case_path)
        logger.info(f"Successfully loaded IEEE 300-bus system with {system.Bus.n} buses")
        
        return system
    
    except Exception as e:
        logger.error(f"Failed to load IEEE 300-bus system: {str(e)}")
        raise

def load_custom_case(case_path):
    """
    Load a custom power system case from a specified file path.
    
    Args:
        case_path (str): Path to the case file
        
    Returns:
        andes.System: ANDES system object with the loaded system
    """
    try:
        logger.info(f"Loading custom case from {case_path}")
        
        # Initialize ANDES system
        system = andes.System()
        system.config.default_config()
        
        # Load the case
        system.load_file(case_path)
        logger.info(f"Successfully loaded custom case with {system.Bus.n} buses")
        
        return system
    
    except Exception as e:
        logger.error(f"Failed to load custom case: {str(e)}")
        raise

def get_system_info(system):
    """
    Get summary information about the system.
    
    Args:
        system (andes.System): ANDES system object
        
    Returns:
        dict: Dictionary containing summary information
    """
    info = {
        'n_buses': system.Bus.n,
        'n_generators': system.GENROU.n if hasattr(system, 'GENROU') else 0,
        'n_lines': system.Line.n,
        'n_transformers': system.XFMR.n if hasattr(system, 'XFMR') else 0,
        'n_loads': system.PQ.n if hasattr(system, 'PQ') else 0,
        'total_gen_capacity_MW': np.sum(system.GENROU.Sn) if hasattr(system, 'GENROU') else 0,
        'total_load_MW': np.sum(system.PQ.p0) if hasattr(system, 'PQ') else 0
    }
    
    return info

def save_system(system, output_path):
    """
    Save the system to a specified file path.
    
    Args:
        system (andes.System): ANDES system object
        output_path (str): Path to save the case
    """
    try:
        logger.info(f"Saving system to {output_path}")
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        system.save(output_path)
        logger.info(f"Successfully saved system to {output_path}")
    
    except Exception as e:
        logger.error(f"Failed to save system: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    ieee68 = load_ieee68()
    ieee68_info = get_system_info(ieee68)
    print("IEEE 68-bus system info:")
    for key, value in ieee68_info.items():
        print(f"  {key}: {value}")
    
    ieee300 = load_ieee300()
    ieee300_info = get_system_info(ieee300)
    print("\nIEEE 300-bus system info:")
    for key, value in ieee300_info.items():
        print(f"  {key}: {value}")