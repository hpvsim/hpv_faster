"""
Modular scenario configuration system for HPV intervention analysis.

This module provides classes and functions for defining vaccination and other 
intervention scenarios in a structured, extensible way.
"""

import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union


@dataclass
class VaccinationConfig:
    """Configuration for vaccination intervention scenarios."""
    
    # Core parameters
    coverage: float = 0.80  # Coverage probability (0-1)
    age_range: Tuple[int, int] = (9, 14)  # Age range for vaccination
    start_year: int = 2025  # Year to start vaccination
    product: str = "nonavalent"  # Vaccine product name
    country: Optional[str] = None  # Country name for country-specific interventions
    
    # Schedule parameters  
    frequency: int = 1  # Years between vaccination campaigns (1=annual, 2=biennial, etc.)
    
    # Metadata
    label: str = "Vaccination"  # Human-readable label
    scenario_id: str = ""  # Unique scenario identifier
    
    def __post_init__(self):
        """Generate scenario ID if not provided."""
        if not self.scenario_id:
            freq_names = {1: "annual", 2: "biennial", 3: "triennial", 
                         4: "quadrennial", 5: "quinquennial", 6: "sextennial",
                         7: "septennial", 8: "octennial", 9: "nonennial", 10: "decennial"}
            freq_name = freq_names.get(self.frequency, f"{self.frequency}yr")
            age_str = f"{self.age_range[0]}-{self.age_range[1]}"
            self.scenario_id = f"vx_{freq_name}_{age_str}"
    
    def generate_interventions(self) -> List[hpv.Intervention]:
        """Generate HPVsim intervention objects based on configuration."""
        interventions = []
        
        # Create vaccine product
        prod = hpv.default_vx(prod_name=self.product)
        
        # Country-specific interventions for 2019-2025
        if self.country and self.country.lower() in ['zambia', 'sierra leone']:
            interventions.extend(self._get_country_specific_interventions(prod))
        else:
            # Fallback to original logic if no country specified
            # 1. Initial catchup campaign for full age range (one-time in start year)
            catchup_vx = hpv.campaign_vx(
                prob=self.coverage,
                years=self.start_year,
                product=prod,
                age_range=(9, 14), #Full age range for catchup
                label="Catchup vx",  # Fixed label for analyzer compatibility
            )
            interventions.append(catchup_vx)
        
    # 2. Ongoing campaigns based on frequency (starts after country-specific period)
        start_year = 2025 + self.frequency
        campaign_years = list(range(start_year, 2061, self.frequency))
        target_age = self.age_range
        for year in campaign_years:
            campaign_vx = hpv.campaign_vx(
                prob=self.coverage,
                years=year,
                product=prod,
                age_range=target_age,
                label="Routine vx",  # Use standard label for analyzer compatibility
            )
            interventions.append(campaign_vx)
        
        return interventions
    
    def _get_country_specific_interventions(self, prod) -> List[hpv.Intervention]:
        """Get country-specific interventions for 2019-2025."""
        interventions = []

        if self.country.lower() == 'zambia':
            interventions_config = [
                (2019, 0.75, 9, 14, "Zambia 2019 vx"),
                (2020, 0.61, 9, 14, "Zambia 2020 vx"),
                (2021, 0.39, 9, 14, "Zambia 2021 vx"),
                (2022, 0.38, 9, 10, "Zambia 2022 vx"),
                (2023, 0.63, 9, 14, "Zambia 2023 vx"),
                (2024, 0.23, 9, 10, "Zambia 2024 vx"),
                (2025, 0.23, 9, 10, "Zambia 2025 vx"),
            ]

        elif self.country.lower() == 'sierra leone':
            # Sierra Leone specific interventions 2022-2025
            interventions_config = [
                # Year, coverage, age_min, age_max, label
                (2023, 0.2, 10, 11, "Sierra Leone 2023 vx"),  # Placeholder - update with actual values
                (2024, 0.43, 10, 11, "Sierra Leone 2024 vx"),  # Placeholder - update with actual values
                (2025, 0.91, 11, 18, "Sierra Leone 2025 vx"),  # Placeholder - update with actual values
            ]
        else:
            interventions_config = []
        
        # Create interventions from configuration
        #second_dose_eligible = lambda sim: (sim.people.vx_doses == 1) | (sim.t > (sim.people.date_vaccinated + 0.5 / sim['dt']))

        for year, coverage, age_min, age_max, label in interventions_config:
            if coverage > 0:  # Only create intervention if coverage > 0
                intervention = hpv.campaign_vx(
                    prob=coverage,
                    years=year,
                    product=prod,
                    age_range=(age_min, age_max),
                    label=label,
                )
                interventions.append(intervention)
        
        return interventions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'coverage': self.coverage,
            'age_range': self.age_range,
            'start_year': self.start_year,
            'product': self.product,
            'country': self.country,
            'frequency': self.frequency,
            'label': self.label,
            'scenario_id': self.scenario_id
        }


@dataclass 
class RiskProfileConfig:
    """Configuration for sexual behavior risk profiles."""
    
    country: str  # Country name
    risk_level: str = "national"  # "national" or "high_risk"
    subregion: Optional[str] = None  # Specific subregion for high-risk
    
    # Available subregions by country
    SUBREGIONS = {
        "zambia": {
            "national": "Zambia",
            "high_risk": ["Zambia High Risk"],
        },
        "sierra leone": {
            "national": "Sierra Leone",
            "high_risk": ["Sierra Leone High Risk"],
        }
    }
    
    def get_sb_pars_name(self) -> str:
        """Get the sb_pars location name for this risk profile."""
        country_lower = self.country.lower()
        
        if self.risk_level == "national":
            return self.SUBREGIONS[country_lower]["national"]
        elif self.risk_level == "high_risk":
            if self.subregion:
                return self.subregion
            else:
                return self.SUBREGIONS[country_lower]["high_risk"][0]
        else:
            raise ValueError(f"Unknown risk level: {self.risk_level}")
    
    def get_available_subregions(self, risk_level: str = "high_risk") -> List[str]:
        """Get list of available subregions for this country and risk level."""
        country_lower = self.country.lower()
        return self.SUBREGIONS.get(country_lower, {}).get(risk_level, [])


@dataclass
class ScenarioDefinition:
    """Complete scenario definition combining intervention and risk profile."""
    
    vaccination: Optional[VaccinationConfig] = None
    risk_profile: Optional[RiskProfileConfig] = None
    # Future: screening, treatment, etc.
    
    # Metadata
    name: str = ""
    description: str = ""
    
    def __post_init__(self):
        """Generate name if not provided."""
        if not self.name and self.vaccination:
            if self.risk_profile and self.risk_profile.risk_level == "national":
                risk_suffix = "NAT"
            else:
                risk_suffix = "HR"
            self.name = f"{self.vaccination.scenario_id}_{risk_suffix}"
    
    def generate_all_interventions(self) -> List[hpv.Intervention]:
        """Generate all intervention objects for this scenario."""
        interventions = []
        
        if self.vaccination:
            interventions.extend(self.vaccination.generate_interventions())
        
        # Future: add screening, treatment, etc.
        
        return interventions
    
    def get_result_filename(self, country: str, filestem: str = "_nov06") -> str:
        """Generate standardized result filename."""
        country_clean = country.lower().replace(" ", "_").replace("'", "_")
        risk_suffix = ""
        if self.risk_profile:
            risk_suffix = f"_{self.risk_profile.risk_level}"
        return f"{country_clean}_{self.name}{risk_suffix}{filestem}.obj"


class ScenarioBuilder:
    """Factory class for building scenario configurations."""
    
    def __init__(self):
        self.countries = ["zambia", "sierra leone"]
        self.base_config = {
            'coverage': 0.80,  # More realistic coverage level
            'start_year': 2025,
            'product': 'nonavalent'
        }
    
    def create_vaccination_scenarios(self) -> List[ScenarioDefinition]:
        """Create all vaccination scenario combinations."""
        scenarios = []
        
        # Define vaccination schedules
        vx_schedules = [
            {"frequency": 1, "age_range": (9, 10), "label": "Annual"},
            {"frequency": 2, "age_range": (9, 11), "label": "Biennial"},
            {"frequency": 3, "age_range": (9, 12), "label": "Triennial"},
            {"frequency": 4, "age_range": (9, 13), "label": "Quadrennial"},
            {"frequency": 5, "age_range": (9, 14), "label": "Quinquennial"},
            {"frequency": 6, "age_range": (9, 15), "label": "Sextennial"},
            {"frequency": 7, "age_range": (9, 16), "label": "Septennial"},
            {"frequency": 8, "age_range": (9, 17), "label": "Octennial"},
            {"frequency": 9, "age_range": (9, 18), "label": "Nonennial"},
            {"frequency": 10, "age_range": (9, 19), "label": "Decennial"},
        ]
        
        # Create scenarios for each country and risk profile
        for country in self.countries:
            for risk_level in ["national", "high_risk"]:
                
                # Baseline (no vaccination)
                risk_config = RiskProfileConfig(country=country, risk_level=risk_level)
                # Use consistent naming with vaccination scenarios
                risk_suffix = "NAT" if risk_level == "national" else "HR"
                baseline = ScenarioDefinition(
                    vaccination=None,
                    risk_profile=risk_config,
                    name=f"baseline_{risk_suffix}",
                    description=f"No vaccination - {country} {risk_level}"
                )
                scenarios.append(baseline)
                
                # Vaccination scenarios
                for schedule in vx_schedules:
                    vx_config = VaccinationConfig(
                        country=country,
                        **{**self.base_config, **schedule}
                    )
                    
                    scenario = ScenarioDefinition(
                        vaccination=vx_config,
                        risk_profile=risk_config,
                        description=f"{schedule['label']} vaccination - {country} {risk_level}"
                    )
                    scenarios.append(scenario)
        
        return scenarios
    
    def create_scenario_matrix(self) -> pd.DataFrame:
        """Create a matrix of all scenario combinations."""
        scenarios = self.create_vaccination_scenarios()
        
        data = []
        for scenario in scenarios:
            row = {
                'scenario_name': scenario.name,
                'country': scenario.risk_profile.country if scenario.risk_profile else "all",
                'risk_level': scenario.risk_profile.risk_level if scenario.risk_profile else "national",
                'intervention_type': "vaccination" if scenario.vaccination else "baseline",
                'coverage': scenario.vaccination.coverage if scenario.vaccination else 0.0,
                'frequency': scenario.vaccination.frequency if scenario.vaccination else None,
                'age_range': str(scenario.vaccination.age_range) if scenario.vaccination else None,
                'description': scenario.description
            }
            data.append(row)
        
        return pd.DataFrame(data)


class ScenarioValidator:
    """Validate scenario configurations for consistency and feasibility."""
    
    @staticmethod
    def validate_vaccination_config(config: VaccinationConfig) -> List[str]:
        """Validate vaccination configuration parameters."""
        issues = []
        
        if not 0 <= config.coverage <= 1:
            issues.append(f"Coverage must be between 0 and 1, got {config.coverage}")
        
        if config.age_range[0] >= config.age_range[1]:
            issues.append(f"Invalid age range: {config.age_range}")
        
        if config.frequency < 1:
            issues.append(f"Frequency must be >= 1 year, got {config.frequency}")
        
        if config.start_year < 2020 or config.start_year > 2060:
            issues.append(f"Start year should be between 2020-2060, got {config.start_year}")
        
        return issues
    
    @staticmethod
    def validate_risk_profile(config: RiskProfileConfig) -> List[str]:
        """Validate risk profile configuration."""
        issues = []
        
        if config.country.lower() not in config.SUBREGIONS:
            issues.append(f"Unknown country: {config.country}")
        
        if config.risk_level not in ["national", "high_risk"]:
            issues.append(f"Risk level must be 'national' or 'high_risk', got {config.risk_level}")
        
        return issues
    
    @staticmethod
    def validate_scenario(scenario: ScenarioDefinition) -> List[str]:
        """Validate complete scenario definition."""
        issues = []
        
        if scenario.vaccination:
            issues.extend(ScenarioValidator.validate_vaccination_config(scenario.vaccination))
        
        if scenario.risk_profile:
            issues.extend(ScenarioValidator.validate_risk_profile(scenario.risk_profile))
        
        if not scenario.name:
            issues.append("Scenario must have a name")
        
        return issues


# Convenience functions for common operations
def create_vaccination_scenario(frequency: int, age_range: Tuple[int, int], 
                              coverage: float = 0.80, country: str = "zambia",
                              risk_level: str = "national") -> ScenarioDefinition:
    """Create a single vaccination scenario with specified parameters."""
    
    vx_config = VaccinationConfig(
        frequency=frequency,
        age_range=age_range,
        coverage=coverage,
        country=country
    )
    
    risk_config = RiskProfileConfig(
        country=country,
        risk_level=risk_level
    )
    
    return ScenarioDefinition(
        vaccination=vx_config,
        risk_profile=risk_config
    )


def load_scenario_from_dict(config_dict: Dict[str, Any]) -> ScenarioDefinition:
    """Load scenario configuration from dictionary."""
    
    vx_config = None
    if 'vaccination' in config_dict:
        vx_config = VaccinationConfig(**config_dict['vaccination'])
    
    risk_config = None
    if 'risk_profile' in config_dict:
        risk_config = RiskProfileConfig(**config_dict['risk_profile'])
    
    return ScenarioDefinition(
        vaccination=vx_config,
        risk_profile=risk_config,
        name=config_dict.get('name', ''),
        description=config_dict.get('description', '')
    )


if __name__ == "__main__":
    # Example usage and testing
    builder = ScenarioBuilder()
    scenarios = builder.create_vaccination_scenarios()
    
    print(f"Generated {len(scenarios)} scenarios")
    print("\nFirst few scenarios:")
    for i, scenario in enumerate(scenarios[:5]):
        print(f"{i+1}. {scenario.name}: {scenario.description}")
    
    # Create scenario matrix
    matrix = builder.create_scenario_matrix()
    print(f"\nScenario matrix shape: {matrix.shape}")
    print(matrix.head())