import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tactile.hand_layout import Pad

def animate_recording(frames, fs, show=True):
    images = np.array([pads_to_image(frame) for frame in frames])
    figure, axes = plt.subplots()
    image = axes.imshow(images[0], cmap="Greens", vmin=0, vmax=4095)
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
    
    if pads.shape != (20, 10, 10):
        raise ValueError(f"Expected input shape (20, 10, 10), given shape: {pads.shape}")
    
    image = np.zeros((40, 50), dtype=pads.dtype)

    # place each of the 20 pads into correct position (approx hand shaped)
    layout = [
        [Pad.THUMB_TIP, Pad.INDEX_TIP, Pad.MIDDLE_TIP, Pad.RING_TIP, Pad.PINKY_TIP],
        [Pad.THUMB_BASE, Pad.INDEX_MIDDLE, Pad.MIDDLE_MIDDLE, Pad.RING_MIDDLE, Pad.PINKY_MIDDLE],
        [Pad.THENAR_UPPER, Pad.INDEX_BASE, Pad.MIDDLE_BASE, Pad.RING_BASE, Pad.PINKY_PALM],
        [Pad.THENAR_INNER, Pad.THENAR_LOWER, Pad.INDEX_PALM, Pad.MIDDLE_PALM, Pad.RING_PALM]
    ]

    for row in range(4):
        for col in range(5):
            pad = layout[row][col]
            image[row*10:(row+1)*10, col*10:(col+1)*10] = pads[pad]

    return image

if __name__ == "__main__":
    frames = np.random.randint(0, 4096, size=(30, 20, 10, 10))
    animate_recording(frames, 10)