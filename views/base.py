from abc import ABC, abstractmethod


class View(ABC):
    """Abstract base class for all views in the application"""
    
    def __init__(self, parent, controller=None):
        self.parent = parent
        self.controller = controller
        self.frame = None
        
    @abstractmethod
    def create_widgets(self):
        """Create and configure the widgets for this view"""
        pass
    
    @abstractmethod
    def show(self):
        """Show this view"""
        pass
    
    @abstractmethod
    def hide(self):
        """Hide this view"""
        pass
    
    def destroy(self):
        """Clean up resources when view is destroyed"""
        if self.frame:
            self.frame.destroy() 