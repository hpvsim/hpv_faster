"""
Risk profile management system for HPV scenario analysis.

This module handles loading and applying different sexual behavior risk profiles
based on geographic regions and population subgroups.
"""

import pandas as pd
import numpy as np
import sciris as sc
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class SexualBehaviorParameters:
    """Container for sexual behavior parameters."""
    
    # Female parameters
    f_dist: str
    f_par1: float
    f_par2: float
    
    # Male parameters  
    m_dist: str
    m_par1: float
    m_par2: float
    
    # Metadata
    location: str
    dist_type: str = "lognormal"
    debut_bias: Tuple[float, float] = (0.0, 0.0)  # (female_bias, male_bias)
    
    def to_hpvsim_format(self) -> Dict[str, Any]:
        """Convert to HPVsim debut parameter format."""
        return {
            'f': {
                'dist': self.f_dist,
                'par1': self.f_par1 + self.debut_bias[0],
                'par2': self.f_par2
            },
            'm': {
                'dist': self.m_dist, 
                'par1': self.m_par1 + self.debut_bias[1],
                'par2': self.m_par2
            }
        }
    
    def copy_with_bias(self, female_bias: float = 0.0, male_bias: float = 0.0) -> 'SexualBehaviorParameters':
        """Create a copy with different debut bias."""
        return SexualBehaviorParameters(
            f_dist=self.f_dist,
            f_par1=self.f_par1,
            f_par2=self.f_par2,
            m_dist=self.m_dist,
            m_par1=self.m_par1,
            m_par2=self.m_par2,
            location=self.location,
            dist_type=self.dist_type,
            debut_bias=(female_bias, male_bias)
        )


class RiskProfileLoader:
    """Loads and manages sexual behavior risk profiles."""
    
    def __init__(self, data_path: str = "../../data", dist_type: str = "lognormal"):
        """
        Initialize the risk profile loader.
        
        Args:
            data_path: Path to directory containing sb_pars CSV files
            dist_type: Distribution type ('lognormal' or 'normal')
        """
        self.data_path = data_path
        self.dist_type = dist_type
        self._sb_data_cache = {}
        self._load_sb_data()
    
    def _load_sb_data(self):
        """Load sexual behavior parameter data from CSV files."""
        try:
            # Load data files
            women_file = f"{self.data_path}/sb_pars_women_{self.dist_type}.csv"
            men_file = f"{self.data_path}/sb_pars_men_{self.dist_type}.csv"
            
            self.sb_data_women = pd.read_csv(women_file)
            self.sb_data_men = pd.read_csv(men_file)
            
            # Create combined lookup
            self._create_parameter_lookup()
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not load sexual behavior data files: {e}. "
                f"Ensure sb_pars files exist in {self.data_path}"
            )
    
    def _create_parameter_lookup(self):
        """Create a lookup dictionary for fast parameter access."""
        self._sb_params = {}
        
        # Get all unique locations
        all_locations = set(self.sb_data_women['location'].unique()) | \
                       set(self.sb_data_men['location'].unique())
        
        for location in all_locations:
            try:
                # Get female parameters
                f_row = self.sb_data_women[self.sb_data_women['location'] == location].iloc[0]
                # Get male parameters
                m_row = self.sb_data_men[self.sb_data_men['location'] == location].iloc[0]
                
                self._sb_params[location] = SexualBehaviorParameters(
                    f_dist=f_row['dist'],
                    f_par1=f_row['par1'],
                    f_par2=f_row['par2'],
                    m_dist=m_row['dist'],
                    m_par1=m_row['par1'],
                    m_par2=m_row['par2'],
                    location=location,
                    dist_type=self.dist_type
                )
            except (IndexError, KeyError) as e:
                print(f"Warning: Could not load parameters for {location}: {e}")
    
    def get_available_locations(self) -> List[str]:
        """Get list of all available parameter locations."""
        return list(self._sb_params.keys())
    
    def get_parameters(self, location: str, debut_bias: Tuple[float, float] = (0.0, 0.0)) -> SexualBehaviorParameters:
        """
        Get sexual behavior parameters for a specific location.
        
        Args:
            location: Location name (e.g., "Zambia", "Zambia Western")
            debut_bias: Bias to apply to debut age (female_bias, male_bias)
            
        Returns:
            SexualBehaviorParameters object
        """
        if location not in self._sb_params:
            available = ", ".join(self.get_available_locations())
            raise ValueError(f"Unknown location '{location}'. Available: {available}")
        
        params = self._sb_params[location]
        if debut_bias != (0.0, 0.0):
            params = params.copy_with_bias(debut_bias[0], debut_bias[1])
        
        return params
    
    def get_country_profiles(self, country: str) -> Dict[str, List[str]]:
        """
        Get available risk profiles for a country.
        
        Args:
            country: Country name
            
        Returns:
            Dictionary with 'national' and 'high_risk' location lists
        """
        country_lower = country.lower()
        
        # Define country mappings
        country_mapping = {
            "zambia": {
                "national": ["Zambia"],
                "high_risk": ["Zambia High Risk"],
            },
            "sierra leone": {
                "national": ["Sierra Leone"],
                "high_risk": ["Sierra Leone High Risk"],
            }
        }
        
        if country_lower not in country_mapping:
            raise ValueError(f"Unknown country: {country}")
        
        # Filter by available locations
        available = self.get_available_locations()
        profiles = {}
        
        for risk_level, locations in country_mapping[country_lower].items():
            profiles[risk_level] = [loc for loc in locations if loc in available]
        
        return profiles
    
    def get_risk_profile_parameters(self, country: str, risk_level: str, 
                                  subregion: Optional[str] = None,
                                  debut_bias: Tuple[float, float] = (0.0, 0.0)) -> SexualBehaviorParameters:
        """
        Get parameters for a specific risk profile.
        
        Args:
            country: Country name
            risk_level: 'national' or 'high_risk'  
            subregion: Specific subregion (for high_risk)
            debut_bias: Bias to apply to debut age
            
        Returns:
            SexualBehaviorParameters object
        """
        profiles = self.get_country_profiles(country)
        
        if risk_level not in profiles:
            raise ValueError(f"Unknown risk level '{risk_level}' for {country}")
        
        available_locations = profiles[risk_level]
        if not available_locations:
            raise ValueError(f"No {risk_level} profiles available for {country}")
        
        # Select location
        if risk_level == "national":
            location = available_locations[0]  # Should only be one
        else:  # high_risk or ultra_high_risk
            if subregion and subregion in available_locations:
                location = subregion
            else:
                location = available_locations[0]  # Default to first available
        
        return self.get_parameters(location, debut_bias)


class RiskProfileManager:
    """High-level manager for risk profile operations."""
    
    def __init__(self, data_path: str = "../../data", dist_type: str = "lognormal"):
        """Initialize with data path and distribution type."""
        self.loader = RiskProfileLoader(data_path, dist_type)
        self.cache = {}
    
    def load_profile_for_scenario(self, country: str, risk_level: str, 
                                 subregion: Optional[str] = None,
                                 debut_bias: Tuple[float, float] = (0.0, 0.0)) -> Dict[str, Any]:
        """
        Load risk profile parameters for a scenario in HPVsim format.
        
        Args:
            country: Country name
            risk_level: 'national' or 'high_risk'
            subregion: Specific subregion for high-risk
            debut_bias: Sexual debut bias
            
        Returns:
            Dictionary in HPVsim debut parameter format
        """
        # Create cache key
        cache_key = (country, risk_level, subregion, debut_bias)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Load parameters
        params = self.loader.get_risk_profile_parameters(
            country, risk_level, subregion, debut_bias
        )
        
        # Convert to HPVsim format
        hpvsim_params = params.to_hpvsim_format()
        
        # Cache result
        self.cache[cache_key] = hpvsim_params
        
        return hpvsim_params
    
    def get_profile_summary(self) -> pd.DataFrame:
        """Get summary of all available risk profiles."""
        data = []
        
        countries = ["zambia", "sierra leone"]
        for country in countries:
            try:
                profiles = self.loader.get_country_profiles(country)
                
                for risk_level, locations in profiles.items():
                    for location in locations:
                        try:
                            params = self.loader.get_parameters(location)
                            data.append({
                                'country': country,
                                'risk_level': risk_level,
                                'location': location,
                                'f_par1': params.f_par1,
                                'f_par2': params.f_par2,
                                'm_par1': params.m_par1,
                                'm_par2': params.m_par2,
                                'dist_type': params.dist_type
                            })
                        except Exception as e:
                            print(f"Warning: Could not load {location}: {e}")
            except Exception as e:
                print(f"Warning: Could not process {country}: {e}")
        
        return pd.DataFrame(data)
    
    def validate_profile_availability(self, country: str, risk_level: str) -> bool:
        """Check if a risk profile is available."""
        try:
            profiles = self.loader.get_country_profiles(country)
            return risk_level in profiles and len(profiles[risk_level]) > 0
        except:
            return False


# Convenience functions
def load_risk_profile(country: str, risk_level: str = "national", 
                     subregion: Optional[str] = None,
                     debut_bias: Tuple[float, float] = (0.0, 0.0),
                     data_path: str = "../../data",
                     dist_type: str = "lognormal") -> Dict[str, Any]:
    """
    Quick function to load a risk profile in HPVsim format.
    
    Args:
        country: Country name  
        risk_level: 'national' or 'high_risk'
        subregion: Specific subregion for high-risk
        debut_bias: Sexual debut bias
        data_path: Path to data directory
        dist_type: Distribution type
        
    Returns:
        Dictionary in HPVsim debut parameter format
    """
    manager = RiskProfileManager(data_path, dist_type)
    return manager.load_profile_for_scenario(country, risk_level, subregion, debut_bias)


def get_available_profiles(data_path: str = "../../data", 
                          dist_type: str = "lognormal") -> pd.DataFrame:
    """Get summary of all available risk profiles."""
    manager = RiskProfileManager(data_path, dist_type)
    return manager.get_profile_summary()


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Risk Profile Management System")
    print("=" * 50)
    
    # Initialize manager
    try:
        manager = RiskProfileManager()
        
        # Get profile summary
        print("\nAvailable Risk Profiles:")
        summary = manager.get_profile_summary()
        print(summary)
        
        # Test loading specific profiles
        print("\nTesting Profile Loading:")
        
        # National profile for Zambia
        zambia_national = manager.load_profile_for_scenario("zambia", "national")
        print(f"Zambia National - Female par1: {zambia_national['f']['par1']:.2f}")
        
        # High-risk profile for Zambia
        zambia_hr = manager.load_profile_for_scenario("zambia", "high_risk")
        print(f"Zambia High Risk - Female par1: {zambia_hr['f']['par1']:.2f}")
        
        # Test with debut bias
        zambia_biased = manager.load_profile_for_scenario("zambia", "national", debut_bias=(-1.0, -1.0))
        print(f"Zambia National (biased) - Female par1: {zambia_biased['f']['par1']:.2f}")
        
        print("\n✅ Risk profile system working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing risk profile system: {e}")
        print("Make sure sb_pars CSV files are available in ../../data/")