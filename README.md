# Power System Stability Analysis with Dynawo

This package provides a comprehensive set of Python modules for power system stability analysis using the Dynawo simulation tool. It enables detailed assessment of transient stability, small-signal stability, voltage stability, and static security for power systems.

## Features

- Load standard IEEE test cases (68-bus and 300-bus)
- Perform topological changes to the network
- Run various types of stability assessments:
  - Static security assessment (power flow, line loading)
  - Small-signal stability analysis (eigenvalues, damping)
  - Voltage stability analysis (time-domain, violations)
  - Transient stability analysis (TSI calculation)
- Inject different types of faults:
  - Line trips
  - Generator trips
  - Bus faults
  - Transformer trips
  - Cascading failures
- Label data based on stability criteria

## Installation

### Prerequisites

- Python 3.8 or higher
- Dynawo simulation tool (see [Dynawo installation guide](https://dynawo.github.io/install/))

### Setup

1. Clone this repository:
   ```bash
    git@github.com:HodgeLab/DataGenerationScriptsPSSA.git
    cd DataGenerationScriptsPSSA
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables for Dynawo:
   ```bash
   export DYNAWO_HOME=/path/to/dynawo
   export DYNAWO_DYD_PATH=/path/to/dyd/files
   export DYNAWO_PAR_PATH=/path/to/par/files
   export DYNAWO_IIDM_PATH=/path/to/iidm/files
   ```

## Modules Overview

### 1. `load_ieee_systems.py`
Loads standard IEEE test systems (68-bus and 300-bus).

```python
from load_ieee_systems import IEEESystemLoader

# Initialize loader
loader = IEEESystemLoader()

# Load IEEE 68-bus system
sim68 = loader.load_ieee68()

# Load IEEE 300-bus system
sim300 = loader.load_ieee300()
```

### 2. `topology_changes.py`
Modifies the topology of power system models.

```python
from topology_changes import TopologyModifier

# Initialize modifier with a simulation
modifier = TopologyModifier(sim68)

# Disconnect a line
modifier.disconnect_line("LINE_1")

# Change line capacity
modifier.change_line_capacity("LINE_2", new_rating=1200)

# Restore original topology
modifier.restore_original_topology()
```

### 3. `static_security.py`
Performs static security assessment through power flows.

```python
from static_security import StaticSecurityAssessor

# Initialize assessor
assessor = StaticSecurityAssessor(sim68)

# Run a power flow
converged = assessor.run_power_flow()

# Calculate overload index
overload_index = assessor.calculate_overload_index()

# Full assessment
results = assessor.assess_static_security()
```

### 4. `small_signal_stability.py`
Analyzes small-signal stability through eigenvalue analysis.

```python
from small_signal_stability import SmallSignalStabilityAnalyzer

# Initialize analyzer
analyzer = SmallSignalStabilityAnalyzer(sim68)

# Linearize system
success = analyzer.linearize_system()

# Compute eigenvalues
eigenvalues = analyzer.compute_eigenvalues()

# Identify oscillatory modes
modes = analyzer.identify_modes(freq_min=0.25, freq_max=1.0)

# Full assessment
results = analyzer.assess_small_signal_stability(min_damping=0.03)
```

### 5. `voltage_stability.py`
Assesses voltage stability through time-domain simulation.

```python
from voltage_stability import VoltageStabilityAnalyzer

# Initialize analyzer
analyzer = VoltageStabilityAnalyzer(sim68)

# Run time-domain simulation
sim_results = analyzer.run_time_domain_simulation()

# Check voltage criteria
criteria_results = analyzer.check_voltage_criteria(
    sim_results, v_min=0.8, v_max=1.1, violation_duration=0.5
)

# Full assessment
results = analyzer.assess_voltage_stability()
```

### 6. `transient_stability.py`
Performs transient stability analysis.

```python
from transient_stability import TransientStabilityAnalyzer

# Initialize analyzer
analyzer = TransientStabilityAnalyzer(sim68)

# Apply a fault
analyzer.apply_fault('line_trip', 'LINE_1', start_time=1.0, duration=0.1)

# Run simulation
sim_results = analyzer.run_simulation(duration=10.0)

# Calculate TSI
tsi = analyzer.calculate_transient_stability_index()

# Full assessment
results = analyzer.assess_transient_stability(tsi_threshold=10)
```

### 7. `fault_injection.py`
Injects various types of faults into the system.

```python
from fault_injection import FaultInjector

# Initialize injector
injector = FaultInjector(sim68)

# Apply line trip
injector.apply_line_trip('LINE_1')

# Apply generator trip
injector.apply_generator_trip('GEN_1')

# Apply cascading failure
injector.apply_cascading_failure()

# Clear all faults
injector.clear_all_faults()
```

### 8. `data_labeling.py`
Labels system states based on stability criteria.

```python
from data_labeling import DataLabeler

# Initialize labeler
labeler = DataLabeler(sim68)

# Assess static security
static_label, static_details = labeler.assess_static_security()

# Assess small signal stability
ss_label, ss_details = labeler.assess_small_signal_stability()

# Assess voltage stability
voltage_label, voltage_details = labeler.assess_voltage_stability()

# Assess transient stability
transient_label, transient_details = labeler.assess_transient_stability()

# Comprehensive assessment
overall_label, overall_details = labeler.assess_overall_stability()

# Get stability DataFrame
df = labeler.get_stability_dataframe()

# Export results
labeler.export_results("stability_results.json")
```

## Stability Criteria

The modules implement the following specific criteria for stability assessment:

1. **Transient Stability**: 
   - System is considered transiently insecure if the TSI is less than 10%
   - TSI = ((360 - delta_max) / (360 + delta_max)) * 100%
   - where delta_max is the maximum angular separation between any two rotor angles

2. **Small-signal Stability**:
   - 3% damping ratio requirement for inter-area oscillation modes
   - Frequency range of interest: 0.25-1.0 Hz

3. **Voltage Stability**:
   - System is insecure if any bus voltage deviates from 0.8-1.1 pu for more than 0.5 seconds

4. **Static Security**:
   - Based on overload index: f_x = sum(w_i * (S_mean,i / S_max,i)^p)
   - Where S_mean,i and S_max,i are the average and maximum apparent power flows

## Complete Example

Here's a complete workflow combining all modules:

```python
from load_ieee_systems import IEEESystemLoader
from fault_injection import FaultInjector
from data_labeling import DataLabeler

# Load system
loader = IEEESystemLoader()
sim = loader.load_ieee68()

# Inject fault
injector = FaultInjector(sim)
injector.apply_line_trip('LINE_1')

# Perform comprehensive assessment
labeler = DataLabeler(sim)
overall_label, overall_details = labeler.assess_overall_stability()

# Export results
results = labeler.export_results()
print(f"Overall stability: {overall_label} - {overall_details}")
```

## Acknowledgments

- [Dynawo](https://dynawo.github.io/) for the simulation framework
- RTE (Réseau de Transport d'Électricité) for developing Dynawo