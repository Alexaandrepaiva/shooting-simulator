import customtkinter as ctk
import os
import logging
from PIL import Image
from .base import View


class StartingMenuView(View):
    """Starting menu view with background image and navigation buttons"""
    
    def __init__(self, parent, controller=None):
        super().__init__(parent, controller)
        self.create_widgets()
        
    def create_widgets(self):
        """Create and configure the starting menu widgets"""
        # Create main frame
        self.frame = ctk.CTkFrame(self.parent)
        
        # Load and set background image
        try:
            # Get the background image path
            bg_image_path = os.path.join("resources", "drawable", "fundo.jpg")
            
            # Load the background image
            bg_image = Image.open(bg_image_path)
            
            # Get screen dimensions for full screen sizing
            screen_width = self.parent.winfo_screenwidth()
            screen_height = self.parent.winfo_screenheight()
            
            # Calculate size to fit screen while maintaining aspect ratio
            img_width, img_height = bg_image.size
            scale_x = screen_width / img_width
            scale_y = screen_height / img_height
            scale = min(scale_x, scale_y)  # Use min so one side reaches the edge
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Create a label with the properly scaled background image
            self.bg_image_label = ctk.CTkLabel(
                self.frame, 
                image=ctk.CTkImage(bg_image, size=(new_width, new_height)),
                text=""
            )
            self.bg_image_label.place(x=0, y=0, relwidth=1, relheight=1)
            
        except Exception as e:
            logging.error(f"Error loading background image: {e}")
            # Create a plain frame if image loading fails
            self.bg_image_label = ctk.CTkLabel(
                self.frame,
                text="Shooting Simulator",
                font=ctk.CTkFont(size=32, weight="bold"),
                text_color="white"
            )
            self.bg_image_label.pack(expand=True)
        
        # Create button container frame
        button_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        button_frame.place(relx=0.5, rely=0.7, anchor="center")
        
        # Green "Iniciar" button
        self.iniciar_button = ctk.CTkButton(
            button_frame,
            text="Iniciar",
            font=ctk.CTkFont(size=20, weight="bold"),
            fg_color="green",
            hover_color="darkgreen",
            text_color="white",
            width=200,
            height=50,
            command=self.on_iniciar_click
        )
        self.iniciar_button.pack(pady=10)
        
        # Blue "Manual de instruções" button
        self.manual_button = ctk.CTkButton(
            button_frame,
            text="Manual de instruções",
            font=ctk.CTkFont(size=20, weight="bold"),
            fg_color="blue",
            hover_color="darkblue",
            text_color="white",
            width=200,
            height=50,
            command=self.on_manual_click
        )
        self.manual_button.pack(pady=10)
        
    def on_iniciar_click(self):
        """Handle Iniciar button click"""
        logging.info("Iniciar button clicked")
        if self.controller:
            self.controller.handle_iniciar()
            
    def on_manual_click(self):
        """Handle Manual button click"""
        logging.info("Manual de Instrucoes button clicked")
        if self.controller:
            self.controller.handle_manual()
    
    def show(self):
        """Show the starting menu view"""
        self.frame.pack(fill="both", expand=True)
        
    def hide(self):
        """Hide the starting menu view"""
        self.frame.pack_forget() 