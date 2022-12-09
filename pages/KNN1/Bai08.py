import tkinter as tk
from PIL import ImageTk, Image

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.resizable(True, True)
        self.image = Image.open('castle.jpg')
        self.canvas = tk.Canvas(self, relief = tk.SUNKEN, borderwidth = 0, bg = 'white', highlightthickness = 0)
        self.canvas.grid(row = 0, column = 0, sticky = tk.NSEW, padx = 5, pady = 5)
        self.canvas.bind("<Configure>", self.configure)        

    def configure(self, event):
        self.canvas.delete('all')
        self.canvas.update()
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        img = self.image.resize((W, H), Image.ANTIALIAS)
        self.image_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor = tk.NW, image = self.image_tk)

if __name__ == "__main__":
    app = App()
    app.title('Bai Thuc Hanh So 8')
    app.mainloop()
