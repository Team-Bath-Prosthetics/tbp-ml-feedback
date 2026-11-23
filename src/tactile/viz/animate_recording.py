import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from tactile import data_format
from tactile.data_format import ADC_MAX, FRAME_SHAPE
from tactile.hand_layout import VIZ_GRID

def animate_recording(frames, fs, name=None, show=True):
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[1:] != FRAME_SHAPE or len(frames) == 0:
        raise ValueError(
            f"animate_recording: expected frames of shape (n_frames, {', '.join(map(str, FRAME_SHAPE))}) "
            f"with n_frames >= 1, given shape: {frames.shape}"
        )

    images = np.array([pads_to_image(frame) for frame in frames])
    n_frames = len(images)

    figure, axes = plt.subplots()
    figure.subplots_adjust(bottom=0.3)
    image = axes.imshow(images[0], cmap="Greens", vmin=0, vmax=ADC_MAX)
    axes.axis("off")
    if name:
        axes.set_title(name)

    '''progress bar which is also a slider that can be dragged to travel through the video'''
    slider_axes = figure.add_axes([0.15, 0.09, 0.7, 0.04])
    slider = Slider(slider_axes, "Frame", 0, max(n_frames - 1, 1), 
                    valinit=0, valstep=1, valfmt="%d")

    button_axes = figure.add_axes([0.42, 0.01, 0.16, 0.06])
    button = Button(button_axes, "Pause")

    state = {"i": 0, "playing": True}

    def frame_generator():
        '''Reads the current position from state, so scrubbing the
        slider makes playback carry on from wherever it was dragged to'''
        while True:
            yield state["i"]
            state["i"] = (state["i"] + 1) % n_frames

    def show_frame(i):
        image.set_data(images[i])
        slider.eventson = False
        slider.set_val(i)
        slider.eventson = True
    
    def update_im(frame):
        show_frame(frame)
        return ()

    def on_slider(value):
        state["i"] = int(value)
        show_frame(state["i"])
        figure.canvas.draw_idle()

    def on_button(_event):
        if state["playing"]:
            heatmap_animation.pause()
            button.label.set_text("Play")
        else:
            heatmap_animation.resume()
            button.label.set_text("Pause")
        state["playing"] = not state["playing"]
        figure.canvas.draw_idle()

    slider.on_changed(on_slider)
    button.on_clicked(on_button)        
    
    heatmap_animation = animation.FuncAnimation(
        figure, update_im, frames=frame_generator, interval=(1000/fs),
        blit=False, cache_frame_data=False
    )

    '''To avoid the widgets being garbage collected (keeping references)'''
    heatmap_animation._widgets = (slider, button)

    if show:
        plt.show()

    return heatmap_animation


def pads_to_image(pads):
    '''takes a 20x10x10 NumPy array (this represents 20 10x10 grids of sensors) and formats them into a 40x50 2D array (4x5 10x10 pads)'''
    
    if pads.shape != FRAME_SHAPE:
        raise ValueError(f"pads_to_image: expected input shape {FRAME_SHAPE}, given shape: {pads.shape}")

    image = np.zeros((40, 50), dtype=pads.dtype)

    # place each of the 20 pads into its position on the (approx hand shaped) grid
    for row, pads_in_row in enumerate(VIZ_GRID):
        for col, pad in enumerate(pads_in_row):
            image[row*10:(row+1)*10, col*10:(col+1)*10] = pads[pad]

    return image

if __name__ == "__main__":
    f = data_format.load_recording("testDs.npz")
    frames = f["frames"]
    print(frames)
    animate_recording(frames, 10, name="testDs")