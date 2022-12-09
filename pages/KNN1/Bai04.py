import tkinter as tk
from PIL import ImageTk, Image

import tensorflow as tf
from tensorflow import keras 
import numpy as np
import cv2
import joblib


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Nhan dang chu so')
        mnist = keras.datasets.mnist
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data() 
        self.X_test = X_test
 
        self.knn = joblib.load("knn_mnist.pkl")

        self.index = None

        self.cvs_digit = tk.Canvas(self, width = 290, height = 290, 
                                   relief = tk.SUNKEN, border = 1)

        self.lbl_ketqua = tk.Label(self, relief = tk.SUNKEN, border = 1, height = 11,
                                         font = ('Consolas', 12))

        btn_create = tk.Button(self, text = 'Create Digit', width = 13, 
                               command = self.btn_create_click)

        btn_recognition = tk.Button(self, text = 'Recognition', width = 13,
                               command = self.btn_recognition_click)

        self.cvs_digit.grid(row = 0, column = 0, padx = 5, pady = 5)
        self.lbl_ketqua.grid(row = 1, column = 0, padx = 5, pady = 5, sticky = tk.EW)

        btn_create.grid(row = 0, column = 1, padx = 5, pady = 5, sticky = tk.N)
        btn_recognition.grid(row = 1, column = 1, padx = 5, pady = 5, sticky = tk.N)

    def btn_create_click(self):
        self.index = np.random.randint(0, 9999, 100)
        digit = np.zeros((10*28,10*28), np.uint8)
        k = 0
        for x in range(0, 10):
            for y in range(0, 10):
                digit[x*28:(x+1)*28, y*28:(y+1)*28] = self.X_test[self.index[k]]
                k = k + 1
        cv2.imwrite('digit.jpg', digit)
        image = Image.open('digit.jpg')
        self.image_tk = ImageTk.PhotoImage(image)
        self.cvs_digit.create_image(0, 0, anchor = tk.NW, image = self.image_tk)
        self.lbl_ketqua.configure(text = '')

    def btn_recognition_click(self):
        sample = np.zeros((100,28,28), np.uint8)
        for i in range(0, 100):
            sample[i] = self.X_test[self.index[i]]

        RESHAPED = 784
        sample = sample.reshape(100, RESHAPED) 
        predicted = self.knn.predict(sample)
        ketqua = ''
        k = 0
        for x in range(0, 10):
            for y in range(0, 10):
                ketqua = ketqua + '%3d' % (predicted[k])
                k = k + 1
            ketqua = ketqua + '\n'
        self.lbl_ketqua.configure(text = ketqua)





if __name__ == "__main__":
    app = App()
    app.mainloop()
