# HPV Vaccination Scenario Analysis Plan

## Overview
This plan outlines the implementation of vaccination-only scenarios for Cote d'Ivoire and Zambia using the _jul21 calibrations, with 30-year follow-up analysis of HPV incidence and cervical cancer incidence.

## Scenario Design

### Core Vaccination Schedules
The scenarios will test different vaccination frequencies and age ranges:

1. **Annual (Every 1 year)**: Ages 9-10
2. **Biennial (Every 2 years)**: Ages 9-11  
3. **Triennial (Every 3 years)**: Ages 9-12
4. **Quadrennial (Every 4 years)**: Ages 9-13
5. **Quinquennial (Every 5 years)**: Ages 9-14
6. **Baseline**: No vaccination

### Risk Profile Variations
Each vaccination schedule will be run with two sexual behavior risk profiles:

1. **National average**: Using country-level sb_pars (Zambia, Cote d'Ivoire)
2. **High-risk regions**: Using subregional sb_pars
   - **Zambia**: Western and Luapula subregions
   - **Cote d'Ivoire**: Denguele subregion

### Future Extensibility: Additional Interventions
Architecture designed to easily accommodate future intervention additions:
- **Screen & treat programs**: Cervical cancer screening with treatment
- **Combined interventions**: Vaccination + screening combinations
- **Multiple coverage levels**: Different programmatic intensities

### Technical Implementation
Each vaccination scenario will use the standard HPVsim pattern:
- **Routine vaccination**: `hpv.routine_vx()` for youngest ages (e.g., age 9-10)
- **Catchup campaigns**: `hpv.campaign_vx()` for remaining age range (e.g., age 10-14)
- **Coverage**: 80% (aligns with standard programmatic assumptions; framework supports 30%, 50%, 70%, 90%)
- **Start year**: 2025 (consistent with existing scenario framework)
- **End year**: 2060 (standard 35-year follow-up, giving 30+ years post-vaccination)
- **Vaccine**: Nonavalent HPV vaccine (default product)
- **Countries**: Cote d'Ivoire and Zambia

## Implementation Steps

### 1. Create Modular Scenario Configuration System
Create `scenario_config.py` with modular architecture:

**Core Components:**
- `VaccinationConfig`: Class for vaccination intervention parameters
- `RiskProfileConfig`: Class for sexual behavior parameter selection
- `ScenarioBuilder`: Factory class to generate complete scenario configurations
- `InterventionRegistry`: Extensible registry for future intervention types

**Key Functions:**
- `get_vaccination_interventions()`: Generate vaccination based on schedule/coverage
- `get_risk_profile_params()`: Select appropriate sb_pars by country/risk level
- `build_scenario_matrix()`: Generate all scenario combinations
- `validate_scenario_config()`: Ensure parameter consistency

### 2. Create Extensible Scenario Runner  
Create `run_hpv_scenarios.py` with extensible architecture:

**Core Features:**
- **Modular intervention loading**: Support vaccination, screening, combined interventions
- **Risk profile integration**: Seamless switching between national/high-risk sb_pars
- **Parameter file management**: Automatic loading of calibrated parameters
- **Configurable outputs**: Flexible analyzer selection based on intervention types

**Architecture:**
```python
class HPVScenarioRunner:
    def __init__(self, config_file):
        self.load_configuration(config_file)
        self.setup_base_parameters()
    
    def run_scenario_batch(self, scenario_list):
        # Parallel execution of scenario list
        
    def setup_intervention_mix(self, vx_config, screen_config=None):
        # Modular intervention combination
        
    def apply_risk_profile(self, location, risk_level):
        # Dynamic sb_pars selection
```

### 3. Configure Outputs and Analyzers
Use standard HPVsim analyzers:
- **DALY analyzer**: `dalys` (starts 2020) - tracks Years of Life Lost and Years Lived with Disability
- **Economic analyzer**: `econ_analyzer` - tracks vaccination counts, cancer cases, and resource utilization
- **Built-in outcomes**: HPVsim automatically tracks `infections`, `cancers`, `cancer_incidence`, `asr_cancer_incidence`, `cancer_deaths`
- **Vaccination tracking**: Number vaccinated by routine vs. catchup programs

### 4. Parameter File Management
**CRITICAL FIRST STEP**: Generate parameter files from calibration files before running scenarios.

The system requires parameter files (e.g., `cote_d_ivoire_pars_jul21.obj`), but we currently have calibration files:
- `cote_d_ivoire_calib_jul21.obj` 
- `zambia_calib_jul21.obj`

**Action needed**: Run `load_calib()` function with `save_pars=True` to extract parameters:
```python
# In run_calibration.py or similar
from run_calibration import load_calib
for location in ['cote d\'ivoire', 'zambia']:
    load_calib(location=location, save_pars=True, filestem="_jul21")
```

This will generate the required `{location}_pars_jul21.obj` files for scenario runs.

## Scenario Matrix

### Core Scenario Structure
| Scenario ID | Description | Vaccination Schedule | Age Range | Risk Profile | Countries |
|-------------|-------------|---------------------|-----------|--------------|-----------|
| V1-0-NAT | Baseline National | None | N/A | National | Both |
| V1-0-HR | Baseline High-Risk | None | N/A | High-Risk | Both |
| V1-1-NAT | Annual National | Every 1 year | 9-10 | National | Both |
| V1-1-HR | Annual High-Risk | Every 1 year | 9-10 | High-Risk | Both |
| V1-2-NAT | Biennial National | Every 2 years | 9-11 | National | Both |
| V1-2-HR | Biennial High-Risk | Every 2 years | 9-11 | High-Risk | Both |
| V1-3-NAT | Triennial National | Every 3 years | 9-12 | National | Both |
| V1-3-HR | Triennial High-Risk | Every 3 years | 9-12 | High-Risk | Both |
| V1-4-NAT | Quadrennial National | Every 4 years | 9-13 | National | Both |
| V1-4-HR | Quadrennial High-Risk | Every 4 years | 9-13 | High-Risk | Both |
| V1-5-NAT | Quinquennial National | Every 5 years | 9-14 | National | Both |
| V1-5-HR | Quinquennial High-Risk | Every 5 years | 9-14 | High-Risk | Both |

**Total scenarios**: 12 scenarios × 2 countries × 2 risk profiles = **48 simulation runs**

### High-Risk Subregion Mapping
- **Zambia**: 
  - National: "Zambia" sb_pars
  - High-Risk: "Zambia Western" and "Zambia Luapula" sb_pars (average or separate runs)
- **Cote d'Ivoire**:
  - National: "Cote d'Ivoire" sb_pars  
  - High-Risk: "Cote d'Ivoire Denguele" sb_pars

## Expected Outputs

### Primary Outcomes
1. **HPV Incidence Trajectories**: Annual new HPV infections (all types, high-risk types, vaccine-preventable types)
2. **Cancer Incidence Trajectories**: Annual new cervical cancer cases
3. **Age-Standardized Incidence Rates**: Comparable epidemiological metrics
4. **Cumulative Impact**: Total cases prevented over 30-year period

### Output Format
- Time series data (annual values 2025-2055)
- Country-specific results
- Scenario comparison matrices
- Confidence intervals from multiple simulation seeds

## Technical Specifications

### Simulation Parameters
- **Population size**: 50,000 agents per simulation
- **Seeds**: 3 (for statistical robustness)
- **Duration**: 2025-2055 (30-year follow-up)
- **Analysis window**: Focus on 2026-2055 (post-vaccination introduction)

### Intervention Parameters
- **Coverage**: 80% for all vaccination scenarios
- **Vaccine efficacy**: As per HPVsim defaults for nonavalent vaccine
- **No other interventions**: Pure vaccination-only analysis

### Computing Requirements
- **Parallel processing**: Can utilize available cores for scenario runs
- **Memory**: Standard HPVsim requirements with shrinking enabled
- **Storage**: Results saved as pickled objects for post-processing

## File Structure

### Modular Architecture for Extensibility
```
/hpv_scenarios/
├── core/
│   ├── __init__.py
│   ├── scenario_config.py              # Modular scenario configuration classes
│   ├── intervention_builder.py         # Vaccination, screening intervention factories  
│   ├── risk_profiles.py               # Sexual behavior parameter management
│   └── base_runner.py                 # Core simulation runner framework
├── configs/
│   ├── vaccination_only_scenarios.yaml # Current vaccination scenario definitions
│   ├── screening_scenarios.yaml        # Future: screening intervention configs
│   └── combined_scenarios.yaml         # Future: vaccination + screening configs
├── runners/
│   ├── run_vaccination_scenarios.py    # Current: vaccination-only runner
│   ├── run_screening_scenarios.py      # Future: screening-only runner
│   └── run_combined_scenarios.py       # Future: combined intervention runner
├── analysis/
│   ├── vaccination_analysis.py         # Vaccination-specific analysis functions
│   ├── screening_analysis.py           # Future: screening analysis functions
│   └── comparative_analysis.py         # Cross-intervention comparison tools
└── results/
    ├── vaccination/
    │   ├── cote_divoire_vaccination_NAT_jul21.obj
    │   ├── cote_divoire_vaccination_HR_jul21.obj
    │   ├── zambia_vaccination_NAT_jul21.obj
    │   └── zambia_vaccination_HR_jul21.obj
    ├── screening/                       # Future: screening results
    └── combined/                        # Future: combined intervention results
```

### Configuration File Examples

**vaccination_only_scenarios.yaml:**
```yaml
base_config:
  start_year: 2025
  end_year: 2060
  coverage: 0.8
  vaccine_product: "nonavalent"

vaccination_schedules:
  annual: {frequency: 1, age_range: [9, 10]}
  biennial: {frequency: 2, age_range: [9, 11]}
  triennial: {frequency: 3, age_range: [9, 12]}
  quadrennial: {frequency: 4, age_range: [9, 13]}
  quinquennial: {frequency: 5, age_range: [9, 14]}

risk_profiles:
  zambia:
    national: "Zambia"
    high_risk: ["Zambia Western", "Zambia Luapula"]
  cote_divoire:
    national: "Cote d'Ivoire" 
    high_risk: ["Cote d'Ivoire Denguele"]
```

## Quality Assurance

### Validation Steps
1. **Baseline verification**: Ensure no-vaccination scenario matches expected epidemiology
2. **Intervention verification**: Confirm vaccination coverage and timing are correctly implemented
3. **Output validation**: Check that HPV and cancer incidence data are being collected properly
4. **Cross-country comparison**: Verify results make epidemiological sense given country differences

### Expected Patterns
- **HPV incidence**: Should decrease with more frequent/broader vaccination
- **Cancer incidence**: Should show delayed decrease (10-20 year lag from HPV prevention)
- **Dose-response**: More comprehensive vaccination should show greater impact
- **Country differences**: Results should reflect underlying epidemiological differences

## Timeline and Dependencies

### Prerequisites
- ✅ _jul21 calibration files exist for both countries
- ⚠️ **Need to generate parameter files** from calibration files (critical first step)
- ✅ HPVsim framework is operational
- ✅ Existing scenario infrastructure can be adapted
- ✅ Countries are defined in `locations.py`
- ✅ Country-specific data files exist in `/data/` directory
- ✅ Subregional sb_pars available for high-risk scenarios:
  - Zambia: Western, Luapula regions
  - Cote d'Ivoire: Denguele region

### Implementation Order
1. **PREREQUISITE**: Generate parameter files from calibration files using `load_calib(save_pars=True)`
2. **Day 1**: Create modular scenario configuration system (`core/scenario_config.py`)
3. **Day 1**: Develop risk profile management system (`core/risk_profiles.py`)
4. **Day 1**: Create vaccination scenario runner (`runners/run_vaccination_scenarios.py`)
5. **Day 1**: Test with debug settings (small populations, single seed, single risk profile)
6. **Day 2**: Full production runs with all risk profile combinations
7. **Day 2**: Results analysis and validation across risk profiles
8. **Future**: Extend framework for screening and combined interventions

## Implementation Issues Identified

### ✅ Completed Fixes (July 24, 2025)
1. **Parameter files**: Generated required `{location}_pars_jul21.obj` files from calibration files
2. **Path resolution**: Fixed file access issues when running from hpv_scenarios subdirectory  
3. **Analyzer compatibility**: Updated `analyzers.py` to handle vaccination-only scenarios
4. **Simulation parameters**: Improved debug parameters for realistic epidemiological patterns
   - Increased population size: 1k → 10k agents in debug mode
   - Maintained high temporal resolution: dt=0.25 even in debug mode
   - Kept full burn-in period: start=1960 even in debug mode
   - Reduced vaccination coverage: 90% → 80% for more realistic scenarios

### ⚠️ Critical Issue: MultiSim Implementation Gap
**PRIORITY FIX NEEDED**: The current vaccination scenario runner does not properly leverage MultiSim capabilities like the main `run_scenarios.py`.

**Current Implementation Problems:**
- Scenarios run with only 1 seed in debug mode (should be 3+ for statistical robustness)
- MultiSim objects are created but not properly structured for uncertainty quantification
- Missing confidence intervals and statistical aggregation across multiple runs
- Results lack the error bounds needed for proper epidemiological interpretation

**Required Changes** (see `run_scenarios.py` as reference):
1. **Multiple Seeds Per Scenario**: Each scenario should run 3+ seeds with different random initializations
2. **Proper MultiSim Structure**: 
   ```python
   # Current: scenarios run individually then combined
   # Needed: scenarios run as seed arrays then properly aggregated
   sims = np.empty((len(scenarios), n_seeds), dtype=object)
   for i_sc, scenario in enumerate(scenarios):
       for i_s in range(n_seeds):
           sims[i_sc, i_s] = create_simulation(scenario, seed=i_s)
   ```
3. **Statistical Aggregation**: Use `make_msims()` function pattern to properly aggregate seeds
4. **Confidence Intervals**: Results should include mean ± confidence intervals from multiple seeds
5. **Robust Comparison**: Statistical significance testing between scenarios

**Impact**: Without proper MultiSim implementation, vaccination scenario comparisons lack statistical rigor and confidence bounds necessary for policy recommendations.

## Risk Mitigation

### Potential Issues
1. ✅ **Missing parameter files**: RESOLVED - Generated required parameter files
2. ✅ **Path issues**: RESOLVED - Fixed directory access from hpv_scenarios
3. ✅ **Analyzer compatibility**: RESOLVED - Updated for vaccination-only scenarios  
4. ✅ **Simulation realism**: RESOLVED - Improved parameters for realistic epidemiology
5. ⚠️ **Statistical robustness**: CRITICAL - MultiSim implementation needs fixing
6. **Memory constraints**: Large population simulations may require optimization
7. **Computational time**: Multiple seeds × scenarios may be time-intensive

### Mitigation Strategies
1. ✅ **Parameter files**: COMPLETED - Generated from calibration files
2. ✅ **Simulation parameters**: COMPLETED - Improved for realistic results
3. ⚠️ **MultiSim implementation**: HIGH PRIORITY - Follow `run_scenarios.py` pattern
4. **Resource monitoring**: Track memory and CPU usage during runs
5. **Incremental testing**: Validate MultiSim changes with small test scenarios first

## Future Extensibility Framework

### Adding New Intervention Types

The modular architecture supports easy addition of new interventions:

**1. Screening & Treat Programs:**
```python
class ScreeningConfig:
    def __init__(self, coverage, age_range, frequency, screen_type):
        self.coverage = coverage          # e.g., 0.7 for 70%
        self.age_range = age_range        # e.g., [25, 65]
        self.frequency = frequency        # e.g., 3 for every 3 years
        self.screen_type = screen_type    # 'cytology', 'hpv', 'via'
        
    def generate_interventions(self):
        return hpv.routine_screening(...)
```

**2. Combined Intervention Scenarios:**
```python
class CombinedScenarioBuilder:
    def __init__(self, vx_config, screen_config):
        self.vaccination = vx_config
        self.screening = screen_config
        
    def build_intervention_mix(self):
        interventions = []
        interventions.extend(self.vaccination.generate_interventions())
        interventions.extend(self.screening.generate_interventions())
        return interventions
```

**3. Intervention Registry Pattern:**
```python
INTERVENTION_REGISTRY = {
    'vaccination': VaccinationConfig,
    'screening': ScreeningConfig,
    'treatment': TreatmentConfig,      # Future
    'education': EducationConfig,      # Future
}

def build_scenario(intervention_specs):
    interventions = []
    for spec in intervention_specs:
        builder_class = INTERVENTION_REGISTRY[spec['type']]
        builder = builder_class(**spec['params'])
        interventions.extend(builder.generate_interventions())
    return interventions
```

### Risk Profile Extensibility

The risk profile system supports:
- **Geographic granularity**: National, regional, district-level parameters
- **Demographic stratification**: Age-specific, socioeconomic risk groups  
- **Behavioral variations**: Different sexual behavior parameter sets
- **Dynamic selection**: Runtime risk profile switching based on scenario needs

### Analysis Framework Extensions

The analysis system can be extended for:
- **Intervention-specific outcomes**: Screen positivity rates, treatment completions
- **Economic evaluations**: Cost-effectiveness across intervention combinations
- **Equity assessments**: Impact differences across risk profiles
- **Sensitivity analyses**: Parameter uncertainty across intervention types

This plan provides a comprehensive framework for implementing vaccination-only scenarios with clear pathways for future intervention additions and risk profile variations.