import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.device_test import (
    list_devices_info,
    friendly_base_name,
    group_devices,
    build_both_groups,
    pick_device_with_both
)


class TestDeviceDetection(unittest.TestCase):
    """Test cases for audio device detection and selection"""

    @patch('pyaudio.PyAudio')
    def test_list_devices_info_success(self, mock_pyaudio):
        """Test successful device listing"""
        # Mock PyAudio instance
        mock_p = Mock()
        mock_pyaudio.return_value = mock_p

        # Mock device info
        mock_device_info = {
            'name': 'Test Device',
            'maxInputChannels': 2,
            'maxOutputChannels': 2,
            'defaultSampleRate': 44100
        }

        mock_p.get_device_count.return_value = 1
        mock_p.get_device_info_by_index.return_value = mock_device_info

        result = list_devices_info()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], 'Test Device')
        self.assertEqual(result[0]['maxInputChannels'], 2)
        self.assertEqual(result[0]['maxOutputChannels'], 2)
        mock_p.terminate.assert_called_once()

    @patch('pyaudio.PyAudio')
    def test_list_devices_info_empty(self, mock_pyaudio):
        """Test device listing when no devices are available"""
        mock_p = Mock()
        mock_pyaudio.return_value = mock_p
        mock_p.get_device_count.return_value = 0

        result = list_devices_info()

        self.assertEqual(result, [])
        mock_p.terminate.assert_called_once()

    def test_friendly_base_name_simple(self):
        """Test friendly name extraction for simple device name"""
        result = friendly_base_name("Headphones (Realtek Audio)")
        self.assertEqual(result, "Headphones")

    def test_friendly_base_name_complex(self):
        """Test friendly name extraction for complex device name"""
        result = friendly_base_name("Speakers (High Definition Audio Device) (2- Microsoft LifeChat LX-3000)")
        self.assertEqual(result, "Microsoft LifeChat LX-3000")

    def test_friendly_base_name_no_parentheses(self):
        """Test friendly name extraction when no parentheses"""
        result = friendly_base_name("Simple Device Name")
        self.assertEqual(result, "Simple Device Name")

    def test_friendly_base_name_driver_suffix(self):
        """Test friendly name extraction with driver suffix"""
        result = friendly_base_name("Device Name @ Driver")
        self.assertEqual(result, "Device Name")

    def test_group_devices_basic(self):
        """Test basic device grouping"""
        devices = [
            {
                'index': 0,
                'name': 'Headphones (Test)',
                'maxInputChannels': 0,
                'maxOutputChannels': 2,
                'defaultSampleRate': 44100
            },
            {
                'index': 1,
                'name': 'Microphone (Test)',
                'maxInputChannels': 1,
                'maxOutputChannels': 0,
                'defaultSampleRate': 44100
            }
        ]

        result = group_devices(devices)

        self.assertIn('headphones', result)
        self.assertIn('microphone', result)
        self.assertEqual(result['headphones']['display'], 'Headphones')
        self.assertEqual(result['microphone']['display'], 'Microphone')

    def test_group_devices_filter_no_channels(self):
        """Test that devices with no input/output channels are filtered out"""
        devices = [
            {
                'index': 0,
                'name': 'Invalid Device',
                'maxInputChannels': 0,
                'maxOutputChannels': 0,
                'defaultSampleRate': 44100
            }
        ]

        result = group_devices(devices)

        self.assertEqual(len(result), 0)

    def test_build_both_groups_success(self):
        """Test building groups that have both input and output"""
        devices = [
            {
                'index': 0,
                'name': 'Combined Device',
                'maxInputChannels': 1,
                'maxOutputChannels': 2,
                'defaultSampleRate': 44100
            }
        ]

        result = build_both_groups(devices)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['display'], 'Combined Device')
        self.assertIn(0, result[0]['input_idxs'])
        self.assertIn(0, result[0]['output_idxs'])
        self.assertEqual(result[0]['rec_in'], 0)
        self.assertEqual(result[0]['rec_out'], 0)

    def test_build_both_groups_no_both(self):
        """Test building groups when no device has both input and output"""
        devices = [
            {
                'index': 0,
                'name': 'Input Only',
                'maxInputChannels': 1,
                'maxOutputChannels': 0,
                'defaultSampleRate': 44100
            },
            {
                'index': 1,
                'name': 'Output Only',
                'maxInputChannels': 0,
                'maxOutputChannels': 2,
                'defaultSampleRate': 44100
            }
        ]

        result = build_both_groups(devices)

        self.assertEqual(len(result), 0)

    @patch('builtins.input')
    @patch('builtins.print')
    @patch('models.device_test.list_devices_info')
    def test_pick_device_with_both_default_selection(self, mock_list_devices, mock_print, mock_input):
        """Test device picking with default selection (Enter key)"""
        # Mock device list
        mock_list_devices.return_value = [
            {
                'index': 0,
                'name': 'Test Device',
                'maxInputChannels': 1,
                'maxOutputChannels': 2,
                'defaultSampleRate': 44100
            }
        ]

        # Mock user input (empty string for default)
        mock_input.return_value = ""

        result = pick_device_with_both()

        self.assertEqual(result, (0, 0, 'Test Device'))

    @patch('builtins.input')
    @patch('builtins.print')
    @patch('models.device_test.list_devices_info')
    def test_pick_device_with_both_manual_selection(self, mock_list_devices, mock_print, mock_input):
        """Test device picking with manual selection"""
        # Mock device list with multiple devices
        mock_list_devices.return_value = [
            {
                'index': 0,
                'name': 'Device 1',
                'maxInputChannels': 1,
                'maxOutputChannels': 2,
                'defaultSampleRate': 44100
            },
            {
                'index': 1,
                'name': 'Device 2',
                'maxInputChannels': 1,
                'maxOutputChannels': 2,
                'defaultSampleRate': 44100
            }
        ]

        # Mock user input (select second device)
        mock_input.return_value = "2"

        result = pick_device_with_both()

        self.assertEqual(result, (1, 1, 'Device 2'))

    @patch('builtins.input')
    @patch('builtins.print')
    @patch('models.device_test.list_devices_info')
    def test_pick_device_with_both_invalid_selection(self, mock_list_devices, mock_print, mock_input):
        """Test device picking with invalid selection"""
        # Mock device list
        mock_list_devices.return_value = [
            {
                'index': 0,
                'name': 'Test Device',
                'maxInputChannels': 1,
                'maxOutputChannels': 2,
                'defaultSampleRate': 44100
            }
        ]

        # Mock invalid user input
        mock_input.return_value = "invalid"

        result = pick_device_with_both()

        self.assertEqual(result, (None, None, None))

    @patch('builtins.input')
    @patch('builtins.print')
    @patch('models.device_test.list_devices_info')
    def test_pick_device_with_both_out_of_range(self, mock_list_devices, mock_print, mock_input):
        """Test device picking with out-of-range selection"""
        # Mock device list
        mock_list_devices.return_value = [
            {
                'index': 0,
                'name': 'Test Device',
                'maxInputChannels': 1,
                'maxOutputChannels': 2,
                'defaultSampleRate': 44100
            }
        ]

        # Mock out-of-range user input
        mock_input.return_value = "5"

        result = pick_device_with_both()

        self.assertEqual(result, (None, None, None))

    @patch('builtins.input')
    @patch('builtins.print')
    @patch('models.device_test.list_devices_info')
    def test_pick_device_with_both_no_devices(self, mock_list_devices, mock_print, mock_input):
        """Test device picking when no suitable devices are found"""
        # Mock empty device list
        mock_list_devices.return_value = []

        result = pick_device_with_both()

        self.assertEqual(result, (None, None, None))


if __name__ == '__main__':
    unittest.main()