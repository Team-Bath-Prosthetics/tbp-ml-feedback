import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

"""  pads_to_image stand-in  """
def pads_to_image(frame):
    pads = np.asarray(frame)
    grid = pads.reshape(4, 5, 10, 10)
    return grid.transpose(0, 2, 1, 3).reshape(40, 50)

def animate_recording(frames, fs):
    images = np.array([pads_to_image(frame) for frame in frames])
    figure, axes = plt.subplots()
    image = axes.imshow(images[0])
    axes.axis("off")

    def update_im(frame):
        image.set_data(images[frame])
        return image,
    
    heatmap_animation = animation.FuncAnimation(figure, update_im, frames=len(images), interval=(1000/fs), blit=True)
    plt.show()

    return heatmap_animation

 