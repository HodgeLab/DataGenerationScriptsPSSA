# Power System Stability Analysis Framework

This framework provides a comprehensive set of tools for performing power system stability analysis using the ANDES (Advanced Network Dynamic Extension for Simulations) package alongside the use of Dynawo package. It includes modules for loading IEEE test systems, modifying network topology, and performing various types of stability assessments.

## Features

- **IEEE System Loading**: Load IEEE 68-bus and IEEE 118-bus test systems
- **Topology Modifications**: Change transmission line capacities, connections, and bus configurations
- **Stability Assessments**:
  - Static Security Assessment through power flow analysis
  - Small-Signal Stability Analysis with damping ratio criteria
  - Voltage Stability Assessment
  - Transient Stability Analysis with TSI calculation
- **Fault Injection**: Apply various fault types including bus faults, line trips, generator trips, and load changes
- **Data Labeling**: Generate labeled datasets based on comprehensive stability criteria

## Installation

1. Clone this repository:
```bash
git clone git@github.com:HodgeLab/DataGenerationScriptsPSSA.git
cd DataGenerationScriptsPSSA
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Modules

The framework consists of the following Python modules:

- `load_ieee_system.py`: Functions for loading IEEE test systems
- `topology_changes.py`: Tools for modifying power system topology
- `static_security.py`: Static security assessment through power flows
- `small_signal_stability.py`: Small-signal stability analysis
- `voltage_stability.py`: Voltage stability assessment
- `transient_stability.py`: Transient stability analysis
- `fault_injection.py`: Tools for injecting different types of faults
- `data_labeling.py`: Data labeling based on stability criteria

## Usage Examples

### Loading a System

```python
import load_ieee_system

# Load IEEE 68-bus system
system = load_ieee_system.load_ieee68()

# Get system information
system_info = load_ieee_system.get_system_info(system)
print("System info:", system_info)
```

### Modifying Topology

```python
import topology_changes

# Modify line capacity
modified_system = topology_changes.modify_line_capacity(system, line_idx=0, new_capacity_mva=150.0)

# Disconnect a line
modified_system = topology_changes.disconnect_line(system, line_idx=5)
```

### Static Security Assessment

```python
import static_security

# Run power flow
converged, results, solved_system = static_security.run_power_flow(system)

# Calculate line loadings
line_loadings = static_security.calculate_line_loadings(solved_system)

# Calculate overload index
overload_idx, line_indices = static_security.calculate_overload_index(solved_system, p=2)
```

### Small-Signal Stability Analysis

```python
import small_signal_stability

# Perform small-signal assessment
assessment = small_signal_stability.perform_small_signal_assessment(system, damping_threshold=3.0)
```

### Transient Stability Analysis

```python
import transient_stability
import fault_injection

# Create a system with a fault
faulted_system = fault_injection.apply_bus_fault(system, bus_idx=0)

# Perform transient stability assessment
assessment = transient_stability.perform_transient_stability_assessment(
    faulted_system, t_end=5.0, tsi_threshold=10.0
)
```

### Generating Labeled Data

```python
import data_labeling

# Initialize the labeler
labeler = data_labeling.StabilityLabeler(output_dir='./stability_data')

# Load a system
labeler.load_system('ieee68')

# Generate a labeled dataset
data = labeler.generate_labeled_dataset(n_scenarios=100)

# Get statistics
stats = labeler.get_stability_statistics()
print(stats)
```

## Stability Criteria

The framework implements the following stability criteria:

- **Transient Stability**: A system is considered transiently insecure if the TSI (Transient Stability Index) is less than 10%. 
  - TSI = (360-δ_max)/(360+δ_max)*100%, where δ_max is the maximum angular separation between any two rotor angles.

- **Small-Signal Stability**: A system is considered small-signal stable if the damping ratio of inter-area oscillation modes (0.25-1.0 Hz) is at least 3%.

- **Voltage Stability**: A system is considered voltage unstable if any bus voltage deviates from the range of 0.8 pu to 1.1 pu for more than 0.5 seconds.

- **Static Security**: The system is statically secure if the overload index is within acceptable limits and there are no overloaded lines.
  - Overload index: f_x = sum(wf_i * (S_mean,i / S_max,i)^p) for all lines

## Dependencies

- ANDES: For power system modeling and simulation
- NumPy and Pandas: For numerical computations and data handling
- Matplotlib: For visualization
- Other scientific Python packages

## References

- Liu, Z., et al. (2018). Accurate Power Swing-Based Algorithm for Transient Stability Assessment.
- Genc, I., et al. (2010). Decision trees for dynamic security assessment.
- Liu, H., et al. (2013). Systematic approach for dynamic security assessment.
- Sevilla, F., et al. (2015). Static security assessment using artificial neural networks.
