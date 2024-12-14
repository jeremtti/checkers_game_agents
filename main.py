import environment as env
import matplotlib.pyplot as plt
import math
from matplotlib.backend_bases import MouseButton

running = True
x, y = [], []
envi = env.Environment()

def on_click(event):
    global x, y, envi

    if event.button is MouseButton.LEFT:
        x.append(event.xdata)
        y.append(event.ydata)

    if event.button is MouseButton.RIGHT:
        valid = True
        for i, j in zip(x, y):
            if i is None or j is None:
                valid = False
        if valid:
            x = [math.floor(i + 0.5) for i in x]
            y = [math.floor(j + 0.5) for j in y]
            square_sequence = [envi.square_id((i, j)) for i, j in zip(x,y)]
            print(square_sequence)
            envi.move(square_sequence)
            plt.imshow(envi.board_image())
            plt.show()
        x, y = [], []

def stop(event):
    global running
    running = False

def show_board(envi):
    global running

    fig = plt.figure()
    _ = fig.canvas.mpl_connect('button_press_event', on_click)
    _ = fig.canvas.mpl_connect('close_event', stop)

    plt.imshow(envi.board_image())
    plt.show()

show_board(envi)