#!/usr/bin/env python3
"""
Shooting Simulator Application

A Python-based shooting simulator that uses webcam to track laser shots
and provides feedback to shooters.

Author: Shooting Simulator Development Team
"""

import sys
import os
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controllers.app_controller import AppController


def main():
    """Main entry point of the application"""
    try:
        # Create and run the application
        app = AppController()
        app.run()
        
    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 