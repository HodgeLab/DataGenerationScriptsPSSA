"""
Module for loading IEEE 68-bus and IEEE 300-bus test systems in Dynawo.
"""

import os
import dynawo
import numpy as np


class IEEESystemLoader:
    """Class to load standard IEEE test systems for Dynawo simulations."""
    
    def __init__(self, dyd_path=None, par_path=None, iidm_path=None):
        """
        Initialize the IEEE system loader.
        
        Args:
            dyd_path: Path to the DYD (Dynawo Dynamic Data) files directory
            par_path: Path to the PAR (Parameters) files directory
            iidm_path: Path to the IIDM (iTesla Internal Data Model) files directory
        """
        self.dyd_path = dyd_path or os.environ.get('DYNAWO_DYD_PATH', './dyd')
        self.par_path = par_path or os.environ.get('DYNAWO_PAR_PATH', './par')
        self.iidm_path = iidm_path or os.environ.get('DYNAWO_IIDM_PATH', './iidm')
        
    def load_ieee68(self, return_data=False):
        """
        Load the IEEE 68-bus (16-machine) system.
        
        Args:
            return_data: If True, returns the loaded system data
            
        Returns:
            A Dynawo simulation object with the IEEE 68-bus system loaded
        """
        print("Loading IEEE 68-bus system...")
        
        # Create a new simulation
        simulation = dynawo.Simulation()
        
        # Define the paths to the necessary files
        iidm_file = os.path.join(self.iidm_path, 'ieee68.iidm')
        dyd_file = os.path.join(self.dyd_path, 'ieee68.dyd')
        par_file = os.path.join(self.par_path, 'ieee68.par')
        
        # Load the network model
        simulation.set_network_data(iidm_file)
        
        # Load the dynamic models
        simulation.set_dynamic_data(dyd_file)
        
        # Load the parameters
        simulation.set_parameters(par_file)
        
        # Set default simulation settings
        simulation.set_time_step(0.01)  # 10ms time step
        simulation.set_duration(20.0)   # 20 seconds simulation
        
        print("IEEE 68-bus system loaded successfully")
        
        if return_data:
            return simulation, self._extract_system_data(simulation)
        return simulation
    
    def load_ieee300(self, return_data=False):
        """
        Load the IEEE 300-bus system.
        
        Args:
            return_data: If True, returns the loaded system data
            
        Returns:
            A Dynawo simulation object with the IEEE 300-bus system loaded
        """
        print("Loading IEEE 300-bus system...")
        
        # Create a new simulation
        simulation = dynawo.Simulation()
        
        # Define the paths to the necessary files
        iidm_file = os.path.join(self.iidm_path, 'ieee300.iidm')
        dyd_file = os.path.join(self.dyd_path, 'ieee300.dyd')
        par_file = os.path.join(self.par_path, 'ieee300.par')
        
        # Load the network model
        simulation.set_network_data(iidm_file)
        
        # Load the dynamic models
        simulation.set_dynamic_data(dyd_file)
        
        # Load the parameters
        simulation.set_parameters(par_file)
        
        # Set default simulation settings
        simulation.set_time_step(0.01)  # 10ms time step
        simulation.set_duration(20.0)   # 20 seconds simulation
        
        print("IEEE 300-bus system loaded successfully")
        
        if return_data:
            return simulation, self._extract_system_data(simulation)
        return simulation
    
    def _extract_system_data(self, simulation):
        """
        Extract key data from the loaded system.
        
        Args:
            simulation: A Dynawo simulation object
            
        Returns:
            Dictionary containing key system data
        """
        network = simulation.get_network()
        
        data = {
            'buses': [],
            'generators': [],
            'loads': [],
            'lines': [],
            'transformers': []
        }
        
        # Extract bus data
        for bus in network.get_buses():
            data['buses'].append({
                'id': bus.get_id(),
                'voltage': bus.get_v(),
                'angle': bus.get_angle(),
                'v_nom': bus.get_v_nom()
            })
        
        # Extract generator data
        for gen in network.get_generators():
            data['generators'].append({
                'id': gen.get_id(),
                'bus_id': gen.get_bus_id(),
                'p': gen.get_p(),
                'q': gen.get_q(),
                'p_max': gen.get_p_max(),
                'q_max': gen.get_q_max(),
                'q_min': gen.get_q_min()
            })
        
        # Extract load data
        for load in network.get_loads():
            data['loads'].append({
                'id': load.get_id(),
                'bus_id': load.get_bus_id(),
                'p': load.get_p(),
                'q': load.get_q()
            })
        
        # Extract line data
        for line in network.get_lines():
            data['lines'].append({
                'id': line.get_id(),
                'bus1_id': line.get_bus1_id(),
                'bus2_id': line.get_bus2_id(),
                'r': line.get_r(),
                'x': line.get_x(),
                'b': line.get_b(),
                'rating': line.get_current_limit()
            })
        
        # Extract transformer data
        for transformer in network.get_transformers():
            data['transformers'].append({
                'id': transformer.get_id(),
                'bus1_id': transformer.get_bus1_id(),
                'bus2_id': transformer.get_bus2_id(),
                'r': transformer.get_r(),
                'x': transformer.get_x(),
                'ratio': transformer.get_ratio()
            })
        
        return data


# Example usage
if __name__ == "__main__":
    loader = IEEESystemLoader()
    
    # Load IEEE 68-bus system
    sim68 = loader.load_ieee68()
    
    # Load IEEE 300-bus system
    sim300 = loader.load_ieee300()