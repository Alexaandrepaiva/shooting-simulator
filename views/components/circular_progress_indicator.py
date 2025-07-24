import customtkinter as ctk
import tkinter as tk
import math

import customtkinter as ctk
import tkinter as tk

class CircularProgressIndicator(ctk.CTkCanvas):
    def __init__(self, master, size=100, arc_length=90, speed=5, **kwargs):
        super().__init__(master, width=size, height=size, highlightthickness=0, **kwargs)

        self.size = size
        self.arc_length = arc_length
        self.speed = speed
        self.angle = 0
        self.arc = None
        self.running = True

        self._draw_circle()
        self.configure(bg="darkgreen")
        self.animate()

    def _draw_circle(self):
        pad = 5
        self.center = self.size // 2
        self.radius = self.center - pad
        self.arc = self.create_arc(
            pad, pad, self.size - pad, self.size - pad,
            start=self.angle,
            extent=self.arc_length,
            style="arc",
            width=5,
            outline="#b4f1b1",
        )

    def animate(self):
        if self.running:
            self.angle = (self.angle + self.speed) % 360
            self.itemconfigure(self.arc, start=self.angle)
            self.after(20, self.animate)

    def stop(self):
        self.running = False

    def start(self):
        if not self.running:
            self.running = True
            self.animate()


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Jetpack-Style Circular Loader")
        self.geometry("300x300")

        self.frame = ctk.CTkFrame(self)
        self.frame.pack(expand=True, fill="both", padx=20, pady=20)

        self.loader = CircularProgressIndicator(self.frame, size=120)
        self.loader.pack(pady=20)

        ctk.CTkButton(self.frame, text="Stop", command=self.loader.stop).pack(pady=5)
        ctk.CTkButton(self.frame, text="Start", command=self.loader.start).pack(pady=5)

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    App().mainloop()
