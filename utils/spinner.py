import customtkinter as ctk
import math
import time
from typing import Optional


class ButtonSpinner:
    """A reusable spinner component for buttons that shows a white rotating animation"""
    
    def __init__(self, button: ctk.CTkButton):
        self.button = button
        self.is_spinning = False
        self.original_text = ""
        self.original_command = None
        self.spinner_after_id = None
        self.rotation_angle = 0
        
    def start_spinner(self):
        """Start the spinner animation and disable the button"""
        if self.is_spinning:
            return
            
        # Store original button state
        self.original_text = self.button.cget("text")
        self.original_command = self.button.cget("command")
        
        # Disable button and change appearance
        self.button.configure(
            text="⟳",  # Unicode spinner character
            command=lambda: None,  # Disable clicking
            state="disabled",
            text_color="white"
        )
        
        self.is_spinning = True
        self.rotation_angle = 0
        self._animate_spinner()
        
    def stop_spinner(self):
        """Stop the spinner animation and restore the button"""
        if not self.is_spinning:
            return
            
        self.is_spinning = False
        
        # Cancel animation
        if self.spinner_after_id:
            try:
                self.button.after_cancel(self.spinner_after_id)
            except:
                pass
            self.spinner_after_id = None
            
        # Restore original button state
        self.button.configure(
            text=self.original_text,
            command=self.original_command,
            state="normal"
        )
        
    def _animate_spinner(self):
        """Animate the spinner rotation"""
        if not self.is_spinning:
            return
            
        # Cycle through different spinner characters for animation effect
        spinner_chars = ["⟳", "⟲", "⟳", "⟲"]
        char_index = (self.rotation_angle // 2) % len(spinner_chars)
        
        self.button.configure(text=spinner_chars[char_index])
        self.rotation_angle = (self.rotation_angle + 1) % 8
        
        # Schedule next animation frame
        self.spinner_after_id = self.button.after(100, self._animate_spinner)


class ButtonStateManager:
    """Manager class to handle button loading states across the application"""
    
    def __init__(self):
        self.button_spinners = {}
        
    def add_button(self, button_id: str, button: ctk.CTkButton):
        """Add a button to be managed"""
        self.button_spinners[button_id] = ButtonSpinner(button)
        
    def start_loading(self, button_id: str):
        """Start loading state for a specific button"""
        if button_id in self.button_spinners:
            self.button_spinners[button_id].start_spinner()
            
    def stop_loading(self, button_id: str):
        """Stop loading state for a specific button"""
        if button_id in self.button_spinners:
            self.button_spinners[button_id].stop_spinner()
            
    def stop_all_loading(self):
        """Stop loading state for all buttons"""
        for spinner in self.button_spinners.values():
            spinner.stop_spinner()
            
    def is_loading(self, button_id: str) -> bool:
        """Check if a button is currently in loading state"""
        if button_id in self.button_spinners:
            return self.button_spinners[button_id].is_spinning
        return False
        
    def cleanup(self):
        """Clean up all spinners"""
        self.stop_all_loading()
        self.button_spinners.clear() 