"""
Module for performing topological changes in Dynawo power system models.
"""

import dynawo
import numpy as np


class TopologyModifier:
    """Class to modify the topology of power system models in Dynawo."""
    
    def __init__(self, simulation):
        """
        Initialize the topology modifier.
        
        Args:
            simulation: A Dynawo simulation object
        """
        self.simulation = simulation
        self.network = simulation.get_network()
        self.original_state = self._save_current_state()
        
    def _save_current_state(self):
        """
        Save the current state of the network topology.
        
        Returns:
            Dictionary containing the current network state
        """
        state = {
            'lines': {},
            'transformers': {},
            'generators': {},
            'buses': {}
        }
        
        # Save line states
        for line in self.network.get_lines():
            line_id = line.get_id()
            state['lines'][line_id] = {
                'status': line.is_connected(),
                'r': line.get_r(),
                'x': line.get_x(),
                'b': line.get_b(),
                'rating': line.get_current_limit()
            }
        
        # Save transformer states
        for transformer in self.network.get_transformers():
            transformer_id = transformer.get_id()
            state['transformers'][transformer_id] = {
                'status': transformer.is_connected(),
                'r': transformer.get_r(),
                'x': transformer.get_x(),
                'ratio': transformer.get_ratio()
            }
        
        # Save generator states
        for generator in self.network.get_generators():
            generator_id = generator.get_id()
            state['generators'][generator_id] = {
                'status': generator.is_connected(),
                'p': generator.get_p(),
                'q': generator.get_q()
            }
        
        # Save bus states
        for bus in self.network.get_buses():
            bus_id = bus.get_id()
            state['buses'][bus_id] = {
                'status': bus.is_connected()
            }
        
        return state
    
    def restore_original_topology(self):
        """
        Restore the network to its original topology.
        """
        print("Restoring original network topology...")
        
        # Restore line states
        for line_id, state in self.original_state['lines'].items():
            line = self.network.get_line(line_id)
            if line.is_connected() != state['status']:
                if state['status']:
                    line.connect()
                else:
                    line.disconnect()
            line.set_r(state['r'])
            line.set_x(state['x'])
            line.set_b(state['b'])
            line.set_current_limit(state['rating'])
        
        # Restore transformer states
        for transformer_id, state in self.original_state['transformers'].items():
            transformer = self.network.get_transformer(transformer_id)
            if transformer.is_connected() != state['status']:
                if state['status']:
                    transformer.connect()
                else:
                    transformer.disconnect()
            transformer.set_r(state['r'])
            transformer.set_x(state['x'])
            transformer.set_ratio(state['ratio'])
        
        # Restore generator states
        for generator_id, state in self.original_state['generators'].items():
            generator = self.network.get_generator(generator_id)
            if generator.is_connected() != state['status']:
                if state['status']:
                    generator.connect()
                else:
                    generator.disconnect()
            generator.set_p(state['p'])
            generator.set_q(state['q'])
        
        # Restore bus states
        for bus_id, state in self.original_state['buses'].items():
            bus = self.network.get_bus(bus_id)
            if bus.is_connected() != state['status']:
                if state['status']:
                    bus.connect()
                else:
                    bus.disconnect()
        
        print("Original topology restored")
    
    def disconnect_line(self, line_id):
        """
        Disconnect a transmission line.
        
        Args:
            line_id: ID of the line to disconnect
        """
        line = self.network.get_line(line_id)
        if line.is_connected():
            print(f"Disconnecting line {line_id}")
            line.disconnect()
        else:
            print(f"Line {line_id} is already disconnected")
    
    def connect_line(self, line_id):
        """
        Connect a transmission line.
        
        Args:
            line_id: ID of the line to connect
        """
        line = self.network.get_line(line_id)
        if not line.is_connected():
            print(f"Connecting line {line_id}")
            line.connect()
        else:
            print(f"Line {line_id} is already connected")
    
    def modify_line_parameters(self, line_id, r=None, x=None, b=None, rating=None):
        """
        Modify parameters of a transmission line.
        
        Args:
            line_id: ID of the line to modify
            r: New resistance value (ohms)
            x: New reactance value (ohms)
            b: New susceptance value (siemens)
            rating: New current rating (amperes)
        """
        line = self.network.get_line(line_id)
        
        if r is not None:
            line.set_r(r)
        
        if x is not None:
            line.set_x(x)
        
        if b is not None:
            line.set_b(b)
        
        if rating is not None:
            line.set_current_limit(rating)
        
        print(f"Modified parameters of line {line_id}")
    
    def disconnect_transformer(self, transformer_id):
        """
        Disconnect a transformer.
        
        Args:
            transformer_id: ID of the transformer to disconnect
        """
        transformer = self.network.get_transformer(transformer_id)
        if transformer.is_connected():
            print(f"Disconnecting transformer {transformer_id}")
            transformer.disconnect()
        else:
            print(f"Transformer {transformer_id} is already disconnected")
    
    def connect_transformer(self, transformer_id):
        """
        Connect a transformer.
        
        Args:
            transformer_id: ID of the transformer to connect
        """
        transformer = self.network.get_transformer(transformer_id)
        if not transformer.is_connected():
            print(f"Connecting transformer {transformer_id}")
            transformer.connect()
        else:
            print(f"Transformer {transformer_id} is already connected")
    
    def change_transformer_tap(self, transformer_id, new_ratio):
        """
        Change the tap ratio of a transformer.
        
        Args:
            transformer_id: ID of the transformer
            new_ratio: New tap ratio value
        """
        transformer = self.network.get_transformer(transformer_id)
        transformer.set_ratio(new_ratio)
        print(f"Changed tap ratio of transformer {transformer_id} to {new_ratio}")
    
    def disconnect_bus(self, bus_id):
        """
        Disconnect a bus.
        
        Args:
            bus_id: ID of the bus to disconnect
        """
        bus = self.network.get_bus(bus_id)
        if bus.is_connected():
            print(f"Disconnecting bus {bus_id}")
            bus.disconnect()
        else:
            print(f"Bus {bus_id} is already disconnected")
    
    def connect_bus(self, bus_id):
        """
        Connect a bus.
        
        Args:
            bus_id: ID of the bus to connect
        """
        bus = self.network.get_bus(bus_id)
        if not bus.is_connected():
            print(f"Connecting bus {bus_id}")
            bus.connect()
        else:
            print(f"Bus {bus_id} is already connected")
    
    def change_line_capacity(self, line_id, new_rating):
        """
        Change the capacity (current rating) of a transmission line.
        
        Args:
            line_id: ID of the line to modify
            new_rating: New current rating (amperes)
        """
        line = self.network.get_line(line_id)
        old_rating = line.get_current_limit()
        line.set_current_limit(new_rating)
        print(f"Changed rating of line {line_id} from {old_rating} A to {new_rating} A")
    
    def get_all_line_ids(self):
        """
        Get IDs of all lines in the network.
        
        Returns:
            List of line IDs
        """
        return [line.get_id() for line in self.network.get_lines()]
    
    def get_all_transformer_ids(self):
        """
        Get IDs of all transformers in the network.
        
        Returns:
            List of transformer IDs
        """
        return [transformer.get_id() for transformer in self.network.get_transformers()]
    
    def get_all_bus_ids(self):
        """
        Get IDs of all buses in the network.
        
        Returns:
            List of bus IDs
        """
        return [bus.get_id() for bus in self.network.get_buses()]


# Example usage
if __name__ == "__main__":
    # This requires a simulation object to be created first
    # from load_ieee_systems import IEEESystemLoader
    # loader = IEEESystemLoader()
    # sim = loader.load_ieee68()
    # topology = TopologyModifier(sim)
    
    # Example operations:
    # line_ids = topology.get_all_line_ids()
    # topology.disconnect_line(line_ids[0])
    # topology.change_line_capacity(line_ids[1], 1000)
    # topology.restore_original_topology()
    pass