import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tactile import data_format
from tactile.data_format import ADC_MAX, FRAME_SHAPE
from tactile.hand_layout import VIZ_GRID

def animate_recording(frames, fs, show=True):
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[1:] != FRAME_SHAPE or len(frames) == 0:
        raise ValueError(
            f"animate_recording: expected frames of shape (n_frames, {', '.join(map(str, FRAME_SHAPE))}) "
            f"with n_frames >= 1, given shape: {frames.shape}"
        )

    images = np.array([pads_to_image(frame) for frame in frames])
    figure, axes = plt.subplots()
    image = axes.imshow(images[0], cmap="Greens", vmin=0, vmax=ADC_MAX)
    axes.axis("off")

    def update_im(frame):
        image.set_data(images[frame])
        return (image,)
    
    heatmap_animation = animation.FuncAnimation(figure, update_im, frames=len(images), interval=(1000/fs), blit=True)
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
    animate_recording(frames, 10)