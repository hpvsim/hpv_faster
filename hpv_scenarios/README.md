# HPV Vaccination Scenarios Analysis Framework

This directory contains a modular framework for running HPV vaccination scenarios with different risk profiles and intervention schedules.

## Overview

The framework is designed to:
- Run vaccination scenarios with different frequencies (annual to quinquennial)
- Compare national vs. high-risk regional populations
- Support extensibility for future interventions (screening, combined programs)
- Generate standardized outputs for analysis

## Directory Structure

```
hpv_scenarios/
├── core/                           # Core framework modules
│   ├── scenario_config.py          # Scenario configuration classes
│   └── risk_profiles.py            # Risk profile management (future)
├── configs/                        # Configuration files
│   └── vaccination_scenarios.yaml  # Scenario definitions (future)
├── runners/                        # Scenario execution scripts
│   └── run_vaccination_scenarios.py # Main runner (future)
├── analysis/                       # Analysis and plotting tools
│   └── vaccination_analysis.py     # Results analysis (future)
├── results/                        # Output storage
│   └── vaccination/                # Vaccination scenario results
└── README.md                       # This file
```

## Core Components

### 1. Scenario Configuration (`core/scenario_config.py`)

The main configuration system provides several key classes:

#### `VaccinationConfig`
Defines vaccination intervention parameters:
- **Coverage**: Vaccination probability (0-1)
- **Age range**: Target ages for vaccination
- **Frequency**: Years between campaigns (1=annual, 2=biennial, etc.)
- **Start year**: When to begin vaccination
- **Product**: Vaccine type (default: nonavalent)

#### `RiskProfileConfig` 
Manages sexual behavior risk profiles:
- **National**: Uses country-level sexual behavior parameters
- **High-risk**: Uses subregional parameters for high-risk areas
  - Zambia: Western and Luapula regions
  - Cote d'Ivoire: Denguele region

#### `ScenarioDefinition`
Complete scenario combining interventions and risk profiles:
- Links vaccination config with risk profile
- Generates HPVsim intervention objects
- Handles result file naming

#### `ScenarioBuilder`
Factory for creating scenario matrices:
- Generates all vaccination schedule combinations
- Creates scenarios for both countries and risk levels
- Produces structured scenario lists and summary matrices

## Vaccination Schedules

The framework tests 5 different vaccination frequencies:

| Schedule | Frequency | Age Range | Description |
|----------|-----------|-----------|-------------|
| Annual | Every 1 year | 9-10 | Routine vaccination + catchup |
| Biennial | Every 2 years | 9-11 | Campaign every 2 years |
| Triennial | Every 3 years | 9-12 | Campaign every 3 years |
| Quadrennial | Every 4 years | 9-13 | Campaign every 4 years |
| Quinquennial | Every 5 years | 9-14 | Campaign every 5 years |

## Risk Profiles

Each vaccination schedule is tested with two risk profiles:

### National Risk Profile
- Uses country-level sexual behavior parameters
- Representative of general population
- Parameter source: "Zambia" or "Cote d'Ivoire" from sb_pars files

### High-Risk Regional Profile  
- Uses parameters from high-risk subregions
- Represents populations with earlier sexual debut/higher risk
- Parameter sources:
  - **Zambia**: "Zambia Western" or "Zambia Luapula" 
  - **Cote d'Ivoire**: "Cote d'Ivoire Denguele"

## Example Usage

### Basic Scenario Creation

```python
from hpv_scenarios.core.scenario_config import *

# Create a single vaccination scenario
scenario = create_vaccination_scenario(
    frequency=2,           # Biennial
    age_range=(9, 11),    # Ages 9-11
    coverage=0.8,         # 80% coverage
    country="zambia",
    risk_level="high_risk"
)

# Generate HPVsim interventions
interventions = scenario.generate_all_interventions()
print(f"Created {len(interventions)} interventions")

# Get result filename
filename = scenario.get_result_filename("zambia", "_jul21")
print(f"Results will be saved as: {filename}")
```

### Generate Full Scenario Matrix

```python
# Create all scenario combinations
builder = ScenarioBuilder()
scenarios = builder.create_vaccination_scenarios()
print(f"Generated {len(scenarios)} scenarios")

# Create summary matrix
matrix = builder.create_scenario_matrix()
print(matrix[['scenario_name', 'country', 'risk_level', 'frequency']])
```

### Scenario Validation

```python
# Validate scenario configuration
validator = ScenarioValidator()
issues = validator.validate_scenario(scenario)
if issues:
    print("Configuration issues:", issues)
else:
    print("Scenario configuration is valid")
```

## Total Scenario Matrix

The framework generates **48 total scenarios**:
- 6 vaccination schedules (baseline + 5 frequencies)
- 2 countries (Zambia, Cote d'Ivoire)
- 2 risk profiles (national, high-risk)
- 6 × 2 × 2 = 24 scenarios per country = **48 total**

## Current Status

### ✅ Completed
- [x] Parameter file generation from calibration files
- [x] Modular scenario configuration system
- [x] Vaccination intervention generation
- [x] Risk profile management system
- [x] Risk profile parameter loading
- [x] Scenario validation
- [x] Documentation framework

### 🚧 In Progress  
- [x] Main scenario runner script
- [ ] Results analysis tools

### 📋 Future Extensions
- [ ] Screening intervention support
- [ ] Combined intervention scenarios
- [ ] Economic analysis integration
- [ ] Visualization tools

## Prerequisites

Before running scenarios, ensure:
1. **Parameter files exist**: `cote_d_ivoire_pars_jul21.obj` and `zambia_pars_jul21.obj` in `../results/`
2. **Sexual behavior parameters**: Updated `sb_pars` files with subregional data
3. **HPVsim environment**: Working HPVsim installation
4. **Data files**: Country-specific epidemiological data in `../data/`

## Risk Profile Management (`core/risk_profiles.py`)

The risk profile system handles loading and applying sexual behavior parameters from different geographic regions.

### Key Classes

#### `RiskProfileLoader`
- Loads sexual behavior parameters from CSV files
- Creates parameter lookup tables for fast access
- Handles parameter validation and error checking

#### `RiskProfileManager`  
- High-level interface for risk profile operations
- Provides caching for performance
- Converts parameters to HPVsim format

#### `SexualBehaviorParameters`
- Container for sexual behavior parameter sets
- Supports debut bias adjustments
- Converts to HPVsim-compatible format

### Available Risk Profiles

Current system supports:

| Country | Risk Level | Location | Description |
|---------|------------|----------|-------------|
| Zambia | National | Zambia | Country-level parameters |
| Zambia | High-Risk | Zambia Western | Western region parameters |
| Zambia | High-Risk | Zambia Luapula | Luapula region parameters |
| Cote d'Ivoire | National | Cote d'Ivoire | Country-level parameters |
| Cote d'Ivoire | High-Risk | Cote d'Ivoire Denguele | Denguele region parameters |

## Quick Tests

Test the configuration system:

```bash
cd /Users/kevinmccarthy/Documents/GitHub/hpvsim_faster/hpv_scenarios/core
python scenario_config.py
```

Test the risk profile system:

```bash  
cd /Users/kevinmccarthy/Documents/GitHub/hpvsim_faster/hpv_scenarios/core
python risk_profiles.py
```

## Main Scenario Runner (`runners/run_vaccination_scenarios.py`)

The main execution script that combines all components to run vaccination scenarios.

### Key Features

- **Parallel execution**: Runs multiple scenarios simultaneously using available CPU cores
- **Automatic result organization**: Creates MultiSim objects with confidence intervals
- **Integrated risk profiles**: Automatically applies appropriate sexual behavior parameters
- **Standardized outputs**: Saves results in organized directory structure
- **Summary reporting**: Generates CSV summaries of key outcomes

### Usage

#### Debug Mode (Quick Test)
```bash
cd /Users/kevinmccarthy/Documents/GitHub/hpvsim_faster/hpv_scenarios/runners
python run_vaccination_scenarios.py
```

This runs with:
- Small population (5,000 agents)
- Single seed per scenario
- Single CPU core
- Limited output

#### Production Mode
Edit the script to set `debug = False` for full analysis:
- Large population (50,000 agents)  
- Multiple seeds (3 per scenario)
- Multi-core parallel processing
- Complete statistical analysis

### Configuration Options

Key parameters in the main() function:

```python
debug = True           # False for production runs
countries = ["zambia", "cote d'ivoire"]
filestem = "_jul21"    # Parameter file suffix
```

### Output Structure

Results are saved to:
```
results/vaccination/
├── zambia_baseline_national_jul21.obj
├── zambia_vx_annual_9-10_NAT_jul21.obj
├── zambia_vx_biennial_9-11_NAT_jul21.obj
├── ...
├── cote_divoire_baseline_national_jul21.obj
├── cote_divoire_vx_annual_9-10_NAT_jul21.obj
└── vaccination_scenarios_summary_jul21.csv
```

## Next Steps

1. **Create risk profile loader** to dynamically load sexual behavior parameters
2. **Build main runner script** to execute scenarios in parallel
3. **Implement results analysis** tools for HPV and cancer incidence tracking
4. **Add debug/test modes** for validation before full runs