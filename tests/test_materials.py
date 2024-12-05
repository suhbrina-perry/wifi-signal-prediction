"""Tests for material properties and signal propagation."""

import unittest
import numpy as np
from absolute.path.src.physics.materials import Material, SignalPath, MATERIALS
from absolute.path.src.data_collection.wifi_data_collector import WiFiDataCollector

class TestMaterials(unittest.TestCase):
    """Test suite for material properties and signal propagation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.concrete = MATERIALS['concrete']
        self.glass = MATERIALS['glass']
        self.collector = WiFiDataCollector()
        
    def test_material_properties(self):
        """Test that material properties are within expected ranges."""
        # Test concrete properties
        self.assertGreater(self.concrete.relative_permittivity, 1)
        self.assertLess(self.concrete.relative_permittivity, 10)
        self.assertGreater(self.concrete.conductivity, 0)
        
        # Test glass properties
        self.assertGreater(self.glass.relative_permittivity, 1)
        self.assertLess(self.glass.conductivity, 0.01)
        
    def test_attenuation_calculation(self):
        """Test that attenuation calculations are physically reasonable."""
        # Test concrete attenuation
        concrete_atten = self.concrete.calculate_attenuation()
        self.assertGreater(concrete_atten, 0)  # Should have positive attenuation
        self.assertLess(concrete_atten, 50)    # Shouldn't be unreasonably high
        
        # Test glass attenuation
        glass_atten = self.glass.calculate_attenuation()
        self.assertGreater(glass_atten, 0)
        self.assertLess(glass_atten, concrete_atten)  # Glass should attenuate less than concrete
        
    def test_signal_path(self):
        """Test signal path calculations through multiple materials."""
        path = SignalPath()
        path.add_layer(self.concrete)
        path.add_layer(self.glass)
        
        total_atten = path.calculate_total_attenuation()
        
        # Total attenuation should be sum of individual attenuations
        expected_atten = (self.concrete.calculate_attenuation() + 
                         self.glass.calculate_attenuation())
        self.assertAlmostEqual(total_atten, expected_atten, places=5)
        
    def test_free_space_loss(self):
        """Test free space path loss calculations."""
        # Test at 1m
        loss_1m = self.collector.calculate_free_space_loss(1.0)
        self.assertGreater(loss_1m, 0)
        
        # Test at 10m
        loss_10m = self.collector.calculate_free_space_loss(10.0)
        self.assertGreater(loss_10m, loss_1m)  # Loss should increase with distance
        
        # Test that it follows inverse square law (approximately)
        # Every doubling of distance should add ~6dB loss
        loss_2m = self.collector.calculate_free_space_loss(2.0)
        self.assertAlmostEqual(loss_2m - loss_1m, 6.0, places=1)
        
    def test_multipath_effects(self):
        """Test multipath effect calculations."""
        original_rssi = -50
        
        # Test with no paths (should return original)
        rssi_no_paths = self.collector.add_multipath_effects(original_rssi, n_paths=0)
        self.assertEqual(rssi_no_paths, original_rssi)
        
        # Test with multiple paths
        rssi_with_paths = self.collector.add_multipath_effects(original_rssi, n_paths=3)
        self.assertNotEqual(rssi_with_paths, original_rssi)  # Should be different
        self.assertGreater(rssi_with_paths, -100)  # Shouldn't be unreasonably low
        
    def test_rssi_calculation(self):
        """Test RSSI calculations with materials."""
        # Test free space
        rssi_free_space = self.collector.calculate_rssi(10.0, None, False)
        self.assertGreater(rssi_free_space, self.collector.noise_floor)
        
        # Test with material
        path = SignalPath()
        path.add_layer(self.concrete)
        rssi_with_material = self.collector.calculate_rssi(10.0, path, False)
        
        # RSSI through material should be lower
        self.assertLess(rssi_with_material, rssi_free_space)
        
        # Test noise floor
        rssi_far = self.collector.calculate_rssi(1000.0)  # Very far distance
        self.assertGreaterEqual(rssi_far, self.collector.noise_floor)

if __name__ == '__main__':
    unittest.main()
