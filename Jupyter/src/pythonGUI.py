from tkinter import *

win = Tk()
win.title("My GUI")
l = Label(win, text="Hello World!!")
b = Button(win, text="ClickMe")
l.pack()
b.pack()
win.mainloop()