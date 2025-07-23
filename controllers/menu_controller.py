import os
import subprocess
import sys
import logging


class MenuController:
    """Controller for the starting menu view"""
    
    def __init__(self, app_controller):
        self.app_controller = app_controller
        self.view = None
        
    def set_view(self, view):
        """Set the view that this controller manages"""
        self.view = view
        
    def handle_iniciar(self):
        """Handle the Iniciar button click"""
        logging.info("Iniciar button clicked")
        # Navigate to calibration view
        self.app_controller.navigate_to_calibration()
        
    def handle_manual(self):
        """Handle the Manual de instruções button click"""
        logging.info("Manual de Instrucoes button clicked")
        self.open_manual_pdf()
        
    def open_manual_pdf(self):
        """Open the manual PDF file"""
        try:
            manual_path = os.path.join("resources", "files", "manual.pdf")
            
            if not os.path.exists(manual_path):
                logging.error(f"Manual PDF not found at: {manual_path}")
                return
                
            # Open PDF with the default system application
            if sys.platform.startswith('win'):
                # Windows
                os.startfile(manual_path)
            elif sys.platform.startswith('darwin'):
                # macOS
                subprocess.run(['open', manual_path])
            else:
                # Linux
                subprocess.run(['xdg-open', manual_path])
                
            logging.info(f"Manual PDF opened: {manual_path}")
            
        except Exception as e:
            logging.error(f"Error opening manual PDF: {e}") 