import customtkinter as ctk
import logging
import cv2
from PIL import Image
from .base import View
from utils.spinner import ButtonStateManager


class SimulationView(View):
    """Simulation view with camera display and recording controls"""
    
    def __init__(self, parent, controller=None):
        super().__init__(parent, controller)
        self.video_label = None
        self.recording_button = None
        self.recalibrate_button = None
        self.is_recording = False
        self.button_state_manager = ButtonStateManager()
        self.create_widgets()
        
    def create_widgets(self):
        """Create and configure the simulation widgets"""
        # Create main frame
        self.frame = ctk.CTkFrame(self.parent)
        
        # Configure grid layout for the main frame
        self.frame.grid_rowconfigure(0, weight=1)  # Video display area
        self.frame.grid_rowconfigure(1, weight=0)  # Buttons area
        self.frame.grid_columnconfigure(0, weight=1)
        
        # Create video display area
        self._create_video_display()
        
        # Create buttons area
        self._create_buttons_area()
        
    def _create_video_display(self):
        """Create the video display area"""
        # Video frame container
        video_frame = ctk.CTkFrame(self.frame)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        video_frame.grid_rowconfigure(0, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)
        
        # Video display label
        self.video_label = ctk.CTkLabel(
            video_frame,
            text="Camera feed will appear here...",
            font=ctk.CTkFont(size=16),
            text_color="gray"
        )
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
    def _create_buttons_area(self):
        """Create the buttons area at the bottom"""
        # Buttons frame
        buttons_frame = ctk.CTkFrame(self.frame, height=80, fg_color="transparent")
        buttons_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        buttons_frame.grid_propagate(False)
        
        # Configure button layout
        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)
        
        # Recording button (Iniciar/Terminar)
        self.recording_button = ctk.CTkButton(
            buttons_frame,
            text="Iniciar",
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#27ae60",  # Green
            hover_color="#229954",
            text_color="white",
            height=50,
            command=self.on_recording_button_click
        )
        self.recording_button.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        # Recalibrate button
        self.recalibrate_button = ctk.CTkButton(
            buttons_frame,
            text="Recalibrar",
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#3498db",  # Blue
            hover_color="#2980b9",
            text_color="white",
            height=50,
            command=self.on_recalibrate_button_click
        )
        self.recalibrate_button.grid(row=0, column=1, sticky="ew", padx=10, pady=10)
        
        # Add buttons to state manager
        self.button_state_manager.add_button("recording", self.recording_button)
        self.button_state_manager.add_button("recalibrate", self.recalibrate_button)
        
    def on_recording_button_click(self):
        """Handle recording button click"""
        if self.button_state_manager.is_loading("recording"):
            return  # Prevent multiple clicks
            
        self.start_recording_loading()
        
        if self.controller:
            if self.is_recording:
                self.controller.stop_recording()
            else:
                self.controller.start_recording()
                
    def on_recalibrate_button_click(self):
        """Handle recalibrate button click"""
        if self.button_state_manager.is_loading("recalibrate"):
            return  # Prevent multiple clicks
            
        logging.info("Recalibrate button clicked")
        self.start_recalibrate_loading()
        
        if self.controller:
            self.controller.handle_recalibrate()
            
    def update_recording_state(self, is_recording):
        """Update the recording button state"""
        self.is_recording = is_recording
        if is_recording:
            self.recording_button.configure(
                text="Terminar",
                fg_color="#e74c3c",  # Red
                hover_color="#c0392b"
            )
        else:
            self.recording_button.configure(
                text="Iniciar",
                fg_color="#27ae60",  # Green
                hover_color="#229954"
            )
            
    def update_video_frame(self, frame_image):
        """Update the video display with a new frame"""
        try:
            if self.video_label and frame_image:
                self.video_label.configure(image=frame_image, text="")
        except Exception as e:
            logging.error(f"Error updating video frame in simulation view: {e}")
            
    def show_no_camera_message(self):
        """Show message when camera is not available"""
        if self.video_label:
            self.video_label.configure(
                text="Câmera não disponível\nVerifique se a câmera está conectada",
                image=None
            )
    
    def start_recording_loading(self):
        """Start loading state for Recording button"""
        self.button_state_manager.start_loading("recording")
        
    def stop_recording_loading(self):
        """Stop loading state for Recording button"""
        self.button_state_manager.stop_loading("recording")
        
    def start_recalibrate_loading(self):
        """Start loading state for Recalibrate button"""
        self.button_state_manager.start_loading("recalibrate")
        
    def stop_recalibrate_loading(self):
        """Stop loading state for Recalibrate button"""
        self.button_state_manager.stop_loading("recalibrate")
        
    def stop_all_loading(self):
        """Stop loading state for all buttons"""
        self.button_state_manager.stop_all_loading()
            
    def show(self):
        """Show the simulation view"""
        self.frame.pack(fill="both", expand=True)
        
    def hide(self):
        """Hide the simulation view"""
        self.frame.pack_forget()
        
    def destroy(self):
        """Clean up resources when view is destroyed"""
        self.button_state_manager.cleanup()
        super().destroy() 