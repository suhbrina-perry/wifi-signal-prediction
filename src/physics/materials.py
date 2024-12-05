"""Module for handling material properties and signal attenuation in WiFi environments."""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass(frozen=True)  # Make the class immutable and hashable
class Material:
    """Class representing a building material and its RF properties."""
    name: str
    relative_permittivity: float  # Dielectric constant (εᵣ)
    conductivity: float          # Electrical conductivity (σ) in S/m
    thickness: float            # Default thickness in meters
    
    def calculate_attenuation(self, frequency: float = 2.4e9) -> float:
        """Calculate signal attenuation through the material.
        
        Uses a simplified plane wave propagation model based on the material's
        electrical properties and thickness.
        
        Args:
            frequency (float): Signal frequency in Hz (default: 2.4 GHz)
            
        Returns:
            float: Attenuation in dB
        """
        # Physical constants
        epsilon_0 = 8.854e-12  # Vacuum permittivity
        mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability
        omega = 2 * np.pi * frequency
        
        # Complex permittivity
        epsilon_r_complex = (self.relative_permittivity - 
                           1j * self.conductivity / (omega * epsilon_0))
        
        # Complex propagation constant
        gamma = 1j * omega * np.sqrt(mu_0 * epsilon_0 * epsilon_r_complex)
        
        # Power attenuation constant (real part of gamma)
        alpha = np.real(gamma)
        
        # Calculate power loss in dB
        attenuation_db = 8.686 * alpha * self.thickness  # 8.686 = 20/ln(10)
        return attenuation_db

# Common building materials with typical properties
MATERIALS = {
    'concrete': Material(
        name='Concrete',
        relative_permittivity=4.5,
        conductivity=0.014,
        thickness=0.2,  # 20cm
    ),
    'glass': Material(
        name='Glass',
        relative_permittivity=6.0,
        conductivity=0.004,
        thickness=0.006,  # 6mm
    ),
    'wood': Material(
        name='Wood',
        relative_permittivity=2.1,
        conductivity=0.002,
        thickness=0.04,  # 4cm
    ),
    'drywall': Material(
        name='Drywall',
        relative_permittivity=2.0,
        conductivity=0.001,
        thickness=0.016,  # 16mm
    ),
    'metal': Material(
        name='Metal',
        relative_permittivity=1.0,
        conductivity=1e7,  # Very high conductivity
        thickness=0.002,  # 2mm
    ),
}

class MaterialLayer:
    """Represents a layer of material in the signal path."""
    def __init__(self, material: Material, thickness_multiplier: float = 1.0):
        """Initialize a material layer.
        
        Args:
            material (Material): The material type
            thickness_multiplier (float): Multiplier for the default thickness
        """
        self.material = material
        self.thickness = material.thickness * thickness_multiplier
        
    def get_attenuation(self, frequency: float = 2.4e9) -> float:
        """Get the total attenuation through this layer.
        
        Args:
            frequency (float): Signal frequency in Hz
            
        Returns:
            float: Attenuation in dB
        """
        return self.material.calculate_attenuation(frequency)

class SignalPath:
    """Represents the path of a signal through various materials."""
    def __init__(self):
        """Initialize an empty signal path."""
        self.layers: List[MaterialLayer] = []
        
    def add_layer(self, material: Material, thickness_multiplier: float = 1.0):
        """Add a material layer to the path.
        
        Args:
            material (Material): The material to add
            thickness_multiplier (float): Multiplier for the default thickness
        """
        self.layers.append(MaterialLayer(material, thickness_multiplier))
        
    def calculate_total_attenuation(self, frequency: float = 2.4e9) -> float:
        """Calculate total attenuation along the path.
        
        Args:
            frequency (float): Signal frequency in Hz
            
        Returns:
            float: Total attenuation in dB
        """
        return sum(layer.get_attenuation(frequency) for layer in self.layers)

# Example usage
if __name__ == "__main__":
    # Create a signal path through multiple materials
    path = SignalPath()
    path.add_layer(MATERIALS['concrete'])  # One concrete wall
    path.add_layer(MATERIALS['drywall'], 2.0)  # Double-thickness drywall
    
    # Calculate total attenuation
    total_loss = path.calculate_total_attenuation()
    print(f"Total signal loss: {total_loss:.1f} dB")
