"""
Test script for the UI integration without requiring all models to be loaded.
This allows you to verify the orb UI state changes work correctly.
"""

import sys
import time
from ui.orb_controller import OrbController

def test_ui_states():
    """Test the orb UI by cycling through all states."""
    
    print("Initializing UI...")
    orb_controller = OrbController()
    orb_controller.start_ui()
    
    # Give UI time to initialize
    time.sleep(2)
    print("UI initialized. Starting state test...\n")
    
    states = [
        ("idle", "Idle state - gentle breathing animation"),
        ("listening", "Listening state - bright, fast pulsing"),
        ("processing", "Processing state - spinning purple arc"),
        ("talking", "Talking state - 3-bar spectrum visualizer"),
    ]
    
    try:
        # Cycle through each state twice
        for cycle in range(2):
            print(f"\n--- Cycle {cycle + 1}/2 ---")
            for state, description in states:
                print(f"Setting state: {state}")
                print(f"  → {description}")
                orb_controller.set_state(state)
                time.sleep(3)  # Show each state for 3 seconds
        
        # Return to idle
        print("\nReturning to idle state...")
        orb_controller.set_state("idle")
        time.sleep(2)
        
        print("\n✓ All state transitions successful!")
        print("The orb will remain visible. Close the window or press Ctrl+C to exit.")
        
        # Keep the program running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nShutting down UI...")
        orb_controller.stop_ui()
        print("✓ UI closed successfully")

if __name__ == "__main__":
    test_ui_states()
