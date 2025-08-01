import customtkinter as ctk
import logging
import cv2
from models.application import ApplicationModel
from views.starting_menu import StartingMenuView
from views.calibration import CalibrationView
from views.simulation import SimulationView
from controllers.menu_controller import StartingMenuController
from controllers.calibration_controller import CalibrationController
from controllers.simulation_controller import SimulationController


class AppController:
    """Main application controller that manages the entire application"""
    
    def __init__(self):
        # Initialize logging
        self.setup_logging()
        
        # Initialize models
        self.app_model = ApplicationModel()
        
        # Camera management - single shared instance
        self.camera = None
        self.is_camera_initialized = False
        
        # Configure GUI appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Shooting Simulator")
        
        # Set full screen
        self.root.state('zoomed')  # Windows full screen
        self.root.resizable(True, True)
        
        # Keep window maximized
        self.root.bind('<Configure>', self._on_window_configure)
        
        # Initialize controllers
        self.starting_menu_controller = StartingMenuController(self)
        self.calibration_controller = CalibrationController(self)
        self.simulation_controller = SimulationController(self)
        
        # Initialize views
        self.starting_menu_view = StartingMenuView(self.root, self.starting_menu_controller)
        self.calibration_view = CalibrationView(self.root, self.calibration_controller)
        self.simulation_view = SimulationView(self.root, self.simulation_controller)
        
        # Set up controller-view relationships
        self.starting_menu_controller.set_view(self.starting_menu_view)
        self.calibration_controller.set_view(self.calibration_view)
        self.simulation_controller.set_view(self.simulation_view)
        
        # Current view tracking
        self.current_view = None
        
        # Start with the starting menu
        self.navigate_to_starting_menu()
        
        logging.info("Application initialized successfully")
        
    def initialize_camera(self):
        """Initialize the shared camera instance"""
        if self.is_camera_initialized and self.camera and self.camera.isOpened():
            return True
            
        try:
            # Release existing camera if any
            if self.camera:
                self.camera.release()
                
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
                
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_camera_initialized = True
            logging.info("Shared camera initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing shared camera: {e}")
            self.camera = None
            self.is_camera_initialized = False
            return False
            
    def get_camera(self):
        """Get the shared camera instance, initializing if needed"""
        if not self.is_camera_initialized:
            self.initialize_camera()
        return self.camera
        
    def is_camera_available(self):
        """Check if the shared camera is available and working"""
        return self.is_camera_initialized and self.camera and self.camera.isOpened()
        
    def release_camera(self):
        """Release the shared camera instance"""
        if self.camera:
            self.camera.release()
            self.camera = None
            self.is_camera_initialized = False
            logging.info("Shared camera released")
        
    def _on_window_configure(self, event):
        """Ensure window stays maximized"""
        try:
            if event.widget == self.root:
                # Keep window maximized
                if self.root.state() != 'zoomed':
                    self.root.state('zoomed')
        except Exception:
            pass
        
    def setup_logging(self):
        """Configure logging for the application"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()  # Console output
            ]
        )
        
    def navigate_to_starting_menu(self):
        """Navigate to the starting menu view"""
        if self.current_view:
            self.current_view.hide()
            
        self.starting_menu_view.show()
        self.current_view = self.starting_menu_view
        self.app_model.set_current_view("starting_menu")
        
        logging.info("Navigated to starting menu")
        
    def navigate_to_calibration(self):
        """Navigate to the calibration view"""
        if self.current_view:
            self.current_view.hide()
            
        self.calibration_view.show()
        self.current_view = self.calibration_view
        self.app_model.set_current_view("calibration")
        
        # Initialize shared camera if needed
        if not self.is_camera_available():
            self.initialize_camera()
        
        # Start calibration process
        self.calibration_controller.start_calibration()
        
        logging.info("Navigated to calibration view")
        
    def navigate_to_simulation(self):
        """Navigate to the simulation view"""
        if self.current_view:
            self.current_view.hide()
            
        self.simulation_view.show()
        self.current_view = self.simulation_view
        self.app_model.set_current_view("simulation")
        
        # Initialize shared camera if needed
        if not self.is_camera_available():
            self.initialize_camera()
        
        # Start simulation process
        self.simulation_controller.start_simulation()
        
        logging.info("Navigated to simulation view")
        
    def navigate_to_shooting_session(self):
        """Navigate to shooting session - to be implemented later"""
        logging.info("Navigate to shooting session - not implemented yet")
        # This will be implemented in future iterations
        pass
        
    def exit_application(self):
        """Clean shutdown of the application"""
        logging.info("Application shutting down")
        
        # Clean up controllers
        if hasattr(self, 'calibration_controller'):
            self.calibration_controller.cleanup()
        if hasattr(self, 'simulation_controller'):
            self.simulation_controller.cleanup()
            
        # Release shared camera
        self.release_camera()
            
        self.root.quit()
        
    def run(self):
        """Start the application main loop"""
        logging.info("Starting application main loop")
        try:
            self.root.mainloop()
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
        finally:
            logging.info("Application terminated") 