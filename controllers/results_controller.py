import json
import os
import cv2
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from models.calibration import CalibrationModel
from models.results import ResultsModel


class ResultsController:
    """Controller for processing detection data and generating results with relative positions"""
    
    def __init__(self):
        self.calibration_model = CalibrationModel()
        self.results_model = ResultsModel()
        
    def process_detection_data(self) -> bool:
        """
        Process detection data and calibration data to generate relative positions
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load calibration data
            if not self.calibration_model.load_calibration_data():
                logging.error("No calibration data found. Cannot process detection data.")
                return False
                
            calibration_data = {
                'parameters': self.calibration_model.get_all_parameters(),
                'fixed_target_positions': self.calibration_model.get_blob_positions()
            }
            
            # Load detection data
            detection_data = self._load_detection_data()
            if not detection_data:
                logging.error("No detection data found. Cannot process.")
                return False
                
            # Process data to generate relative positions
            results_data = self.results_model.calculate_relative_positions(
                detection_data, calibration_data
            )
            
            # Save results data
            success = self.results_model.save_results_data(results_data)
            if success:
                logging.info("Detection data processed successfully and results saved.")
                return True
            else:
                logging.error("Failed to save results data.")
                return False
                
        except Exception as e:
            logging.error(f"Error processing detection data: {e}")
            return False
            
    def generate_result_images(self) -> bool:
        """
        Generate result images with labeled shots using alvobase image and results data
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load results data
            results_data = self.results_model.load_results_data()
            if not results_data:
                logging.error("No results data found. Run process-data first.")
                return False
                
            # Generate images for each target
            success = self.results_model.generate_target_images(results_data)
            if success:
                logging.info("Result images generated successfully.")
                return True
            else:
                logging.error("Failed to generate result images.")
                return False
                
        except Exception as e:
            logging.error(f"Error generating result images: {e}")
            return False
            
    def _load_detection_data(self) -> Optional[Dict]:
        """Load detection data from JSON file"""
        try:
            detection_file = os.path.join("data", "detection_data.json")
            if not os.path.exists(detection_file):
                return None
                
            with open(detection_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logging.error(f"Error loading detection data: {e}")
            return None
            
    def get_processing_status(self) -> Dict[str, Any]:
        """Get status of data processing pipeline"""
        status = {
            "calibration_data_exists": False,
            "detection_data_exists": False,
            "results_data_exists": False,
            "result_images_exist": False
        }
        
        try:
            # Check calibration data
            status["calibration_data_exists"] = os.path.exists(
                os.path.join("data", "calibration_data.json")
            )
            
            # Check detection data
            status["detection_data_exists"] = os.path.exists(
                os.path.join("data", "detection_data.json")
            )
            
            # Check results data
            status["results_data_exists"] = os.path.exists(
                os.path.join("data", "results_data.json")
            )
            
            # Check result images (look for any target result images)
            result_images = []
            if os.path.exists("data"):
                for file in os.listdir("data"):
                    if file.startswith("target_") and file.endswith("_result.jpg"):
                        result_images.append(file)
            status["result_images_exist"] = len(result_images) > 0
            status["result_images"] = result_images
            
        except Exception as e:
            logging.error(f"Error checking processing status: {e}")
            
        return status 