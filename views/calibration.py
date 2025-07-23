import customtkinter as ctk
import logging
import cv2
from PIL import Image
from .base import View
from utils.spinner import ButtonStateManager


class CalibrationView(View):
    """Calibration view with camera display and parameter controls"""
    
    def __init__(self, parent, controller=None):
        super().__init__(parent, controller)
        self.video_label = None
        self.value_labels = {}
        self.dropdown_controls = {}
        self.button_state_manager = ButtonStateManager()
        self.create_widgets()
        
    def create_widgets(self):
        """Create and configure the calibration widgets"""
        # Create main frame
        self.frame = ctk.CTkFrame(self.parent)
        
        # Create left panel for controls (wider and more compact)
        self.left_panel = ctk.CTkFrame(self.frame, width=400)
        self.left_panel.pack(side="left", fill="y", padx=5, pady=5)
        self.left_panel.pack_propagate(False)
        
        # Create right panel for video display
        self.right_panel = ctk.CTkFrame(self.frame)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        self._create_parameter_controls()
        self._create_video_display()
        
    def _create_parameter_controls(self):
        """Create parameter control widgets on the left panel"""
        # Setup variables with exact ranges from original code
        self._setup_variables()
        
        # Title
        title_label = ctk.CTkLabel(
            self.left_panel,
            text="PARÂMETROS DE CALIBRAÇÃO",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#2196F3"
        )
        title_label.pack(pady=(10, 15))
        
        # Create sliders
        self._create_sliders()
        
        # Create dropdowns
        self._create_dropdowns()
        
        # Save button
        self._create_save_button()
        
    def _setup_variables(self):
        """Setup variables with default values matching original code"""
        self.min_area_var = ctk.IntVar(value=30)
        self.max_area_var = ctk.IntVar(value=400)
        self.min_circularity_var = ctk.DoubleVar(value=50)  # 50% = 0.5
        self.shots_per_series_var = ctk.IntVar(value=3)
        self.number_of_targets_var = ctk.IntVar(value=1)
        
        # Variables are already initialized in __init__
        
    def _create_sliders(self):
        """Create sliders with exact ranges from original code"""
        slider_configs = [
            ("Min Area", self.min_area_var, 7, 200),
            ("Max Area", self.max_area_var, self.min_area_var.get(), 1000),
            ("Min Circularity", self.min_circularity_var, 1, 100)
        ]
        
        for label_text, variable, from_, to in slider_configs:
            self._create_slider(label_text, variable, from_, to)
            
    def _create_slider(self, label_text, variable, from_, to):
        """Create a slider matching the original design"""
        # Create frame for slider and its labels (more compact)
        slider_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        slider_frame.pack(pady=(5, 8), padx=8, fill="x")
        
        # Frame for title and value on the same line (more compact)
        header_frame = ctk.CTkFrame(slider_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 3))
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_columnconfigure(1, weight=0)
        
        # Label for parameter name (smaller font)
        name_label = ctk.CTkLabel(
            header_frame, 
            text=label_text, 
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        name_label.grid(row=0, column=0, sticky="w")
        
        # Label for current value (smaller and more compact)
        value_label = ctk.CTkLabel(
            header_frame,
            text=self._format_value(label_text, variable.get()),
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#4CAF50",
            anchor="e",
            fg_color="#ffffff",
            corner_radius=4,
            width=120,
            height=20
        )
        value_label.grid(row=0, column=1, sticky="e", padx=(8, 0))
        
        # Store reference to value label
        self.value_labels[label_text] = value_label
        
        # Slider (more compact)
        slider = ctk.CTkSlider(
            slider_frame,
            from_=from_,
            to=to,
            variable=variable,
            height=16,
            button_color="#255527",
            button_hover_color="#49bb59",
            command=lambda value: self._update_value_display(label_text, value)
        )
        slider.pack(pady=(0, 1), padx=8, fill="x")
        
        # Range info (smaller)
        range_label = ctk.CTkLabel(
            slider_frame,
            text=f"Range: {self._format_range_value(label_text, from_)} - {self._format_range_value(label_text, to)}",
            font=ctk.CTkFont(size=9),
            text_color="#666666"
        )
        range_label.pack(pady=(0, 3))
        
    def _create_video_display(self):
        """Create video display area on the right panel"""
        # Video title
        video_title = ctk.CTkLabel(
            self.right_panel,
            text="Visualização da Câmera",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        video_title.pack(pady=10)
        
        # Video display label (larger to fill space better)
        self.video_label = ctk.CTkLabel(
            self.right_panel,
            text="Aguardando câmera...",
            width=800,
            height=600
        )
        self.video_label.pack(expand=True, padx=10, pady=10)
        
    def _create_dropdowns(self):
        """Create dropdowns with exact ranges from original code"""
        dropdown_configs = [
            ("Tiros por serie", self.shots_per_series_var, list(range(1, 16))),  # 1-15
            ("Numero de Alvos", self.number_of_targets_var, list(range(1, 6)))   # 1-5
        ]
        
        for label_text, variable, values in dropdown_configs:
            self._create_dropdown(label_text, variable, values)
            
    def _create_dropdown(self, label_text, variable, values):
        """Create a dropdown matching the original design"""
        # Create frame to organize dropdown and its labels (more compact)
        dropdown_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        dropdown_frame.pack(pady=(5, 8), padx=8, fill="x")
        
        # Label for parameter name (smaller)
        name_label = ctk.CTkLabel(
            dropdown_frame, 
            text=label_text, 
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        name_label.pack(anchor="w", pady=(0, 3))
        
        # Dropdown (ComboBox) - more compact
        dropdown = ctk.CTkComboBox(
            dropdown_frame,
            values=[str(val) for val in values],
            variable=ctk.StringVar(value=str(variable.get())),
            command=lambda choice: self._on_dropdown_change(label_text, choice, variable),
            height=28,
            font=ctk.CTkFont(size=12, weight="bold"),
            dropdown_font=ctk.CTkFont(size=10),
            button_color="#255527",
            button_hover_color="#49bb59",
            border_color="#255527",
            border_width=2,
            corner_radius=6,
            state="readonly"
        )
        dropdown.pack(pady=(0, 3), padx=8, fill="x")
        
        # Store reference to dropdown
        self.dropdown_controls[label_text] = dropdown
        
        # Info about available options (smaller)
        info_label = ctk.CTkLabel(
            dropdown_frame,
            text=f"Opções: {values[0]} - {values[-1]}",
            font=ctk.CTkFont(size=9),
            text_color="#666666"
        )
        info_label.pack(pady=(0, 3))
        
    def _on_dropdown_change(self, label_text, choice, variable):
        """Callback when a dropdown is changed"""
        try:
            value = int(choice)
            variable.set(value)
            # Notify controller of parameter change
            if self.controller:
                param_name = "shots_per_series" if "Tiros" in label_text else "number_of_targets"
                self.controller.on_parameter_change(param_name, value)
        except ValueError:
            pass
            
    def _create_save_button(self):
        """Create save button matching original design"""
        # Visual separator (smaller)
        separator = ctk.CTkFrame(self.left_panel, height=1, fg_color="#cccccc")
        separator.pack(pady=(10, 8), padx=15, fill="x")
        
        # Informative label (more compact)
        info_label = ctk.CTkLabel(
            self.left_panel,
            text="✓ Ajuste os valores até que os alvos fiquem destacados em VERDE",
            font=ctk.CTkFont(size=10),
            text_color="#666666",
            wraplength=350
        )
        info_label.pack(pady=(0, 10), padx=8)
        
        # Styled save button (more compact)
        self.save_button = ctk.CTkButton(
            self.left_panel,
            text="💾 Salvar Parâmetros",
            command=self.on_save_parameters,
            height=38,
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8,
            fg_color="#27ae60",
            hover_color="#229954"
        )
        self.save_button.pack(pady=(0, 15), padx=15, fill="x")
        
        # Add save button to state manager
        self.button_state_manager.add_button("save", self.save_button)
        
        # Initialize value displays
        self._initialize_value_displays()
        
    def _format_value(self, label_text, value):
        """Format values exactly like original code"""
        if "Circularity" in label_text:
            # Circularity is stored as 0-100, but represents 0.00-1.00
            percentage = value
            return f"{percentage:.0f}% ({value/100:.2f})"
        elif "Area" in label_text:
            return f"{int(value)} px"
        elif "Tiros" in label_text:
            return f"{int(value)} tiros"
        elif "Alvos" in label_text:
            return f"{int(value)} alvos"
        elif isinstance(value, float):
            return f"{value:.1f}"
        else:
            return f"{int(value)}"
            
    def _format_range_value(self, label_text, value):
        """Format range values like original code"""
        if "Circularity" in label_text:
            return f"{int(value)}%"
        else:
            return f"{int(value)}"
            
    def _update_value_display(self, label_text, value):
        """Update value display when slider is moved"""
        if label_text in self.value_labels:
            formatted_value = self._format_value(label_text, value)
            self.value_labels[label_text].configure(text=formatted_value)
            
        # Notify controller of parameter change
        if self.controller:
            param_name = self._get_param_name_from_label(label_text)
            if param_name:
                actual_value = value / 100 if "Circularity" in label_text else value
                self.controller.on_parameter_change(param_name, actual_value)
                
    def _initialize_value_displays(self):
        """Initialize value displays with current values"""
        # Initialize slider displays
        slider_configs = [
            ("Min Area", self.min_area_var),
            ("Max Area", self.max_area_var),
            ("Min Circularity", self.min_circularity_var)
        ]
        
        for label_text, variable in slider_configs:
            if label_text in self.value_labels:
                current_value = variable.get()
                formatted_value = self._format_value(label_text, current_value)
                self.value_labels[label_text].configure(text=formatted_value)
        
        # Initialize dropdown values
        dropdown_configs = [
            ("Tiros por serie", self.shots_per_series_var),
            ("Numero de Alvos", self.number_of_targets_var)
        ]
        
        for label_text, variable in dropdown_configs:
            if label_text in self.dropdown_controls:
                current_value = variable.get()
                self.dropdown_controls[label_text].set(str(current_value))
                
    def _get_param_name_from_label(self, label_text):
        """Convert label text to parameter name"""
        if "Min Area" in label_text:
            return "min_area"
        elif "Max Area" in label_text:
            return "max_area"
        elif "Circularity" in label_text:
            return "min_circularity"
        elif "Tiros" in label_text:
            return "shots_per_series"
        elif "Alvos" in label_text:
            return "number_of_targets"
        return None
            
    def update_video_frame(self, frame_image):
        """Update the video display with a new frame"""
        try:
            if self.video_label:
                self.video_label.configure(image=frame_image, text="")
        except Exception as e:
            logging.error(f"Error updating video frame: {e}")
            
    def get_parameter_values(self):
        """Get current values exactly like original code"""
        return {
            "min_area": self.min_area_var.get(),
            "max_area": self.max_area_var.get(),
            "min_circularity": self.min_circularity_var.get() / 100,  # Convert percentage to decimal
            "shots_per_series": self.shots_per_series_var.get(),
            "number_of_targets": self.number_of_targets_var.get()
        }
        
    def set_parameter_values(self, values):
        """Set parameter values from a dictionary"""
        if "min_area" in values:
            self.min_area_var.set(values["min_area"])
        if "max_area" in values:
            self.max_area_var.set(values["max_area"])
        if "min_circularity" in values:
            self.min_circularity_var.set(values["min_circularity"] * 100)  # Convert decimal to percentage
        if "shots_per_series" in values:
            self.shots_per_series_var.set(values["shots_per_series"])
        if "number_of_targets" in values:
            self.number_of_targets_var.set(values["number_of_targets"])
            
        # Update displays
        self._initialize_value_displays()
                    
    def on_save_parameters(self):
        """Handle save parameters button click"""
        if self.button_state_manager.is_loading("save"):
            return  # Prevent multiple clicks
            
        self.start_save_loading()
        
        if self.controller:
            self.controller.save_parameters()
    
    def start_save_loading(self):
        """Start loading state for Save button"""
        self.button_state_manager.start_loading("save")
        
    def stop_save_loading(self):
        """Stop loading state for Save button"""
        self.button_state_manager.stop_loading("save")
        
    def stop_all_loading(self):
        """Stop loading state for all buttons"""
        self.button_state_manager.stop_all_loading()
    
    def show(self):
        """Show the calibration view"""
        self.frame.pack(fill="both", expand=True)
        
    def hide(self):
        """Hide the calibration view"""
        self.frame.pack_forget()
        
    def destroy(self):
        """Clean up resources when view is destroyed"""
        self.button_state_manager.cleanup()
        super().destroy() 