from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

"""  pads_to_image stand-in  """
def pads_to_image(frame):
    pads = np.asarray(frame)
    grid = pads.reshape(4, 5, 10, 10)
    return grid.transpose(0, 2, 1, 3).reshape(40, 50)

def animate_recording(frames, fs): # This shows 'purple screen'
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


def visualize_pad_over_time(frames, fs, pad, adc_max=4095, save_path=None, show=True):
    """Animate one pad as a heatmap alongside pressure changing over time."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[1:] != (20, 10, 10):
        raise ValueError("frames must have shape (T, 20, 10, 10)")
    if not isinstance(pad, int):
        pad = int(pad)
    if pad < 0 or pad >= 20:
        raise ValueError("pad must be between 0 and 19")
    if not isinstance(fs, (int, float)) or fs <= 0:
        raise ValueError("fs must be a positive sampling rate")

    pad_frames = frames[:, pad]
    times = np.arange(len(frames), dtype=np.float64) / float(fs)
    maximum_pressure = pad_frames.max(axis=(1, 2))
    mean_pressure = pad_frames.mean(axis=(1, 2))

    figure, (heatmap_axes, plot_axes) = plt.subplots(
        1,
        2,
        figsize=(11, 4),
        gridspec_kw={"width_ratios": [1, 1.35]},
    )

    heatmap = heatmap_axes.imshow(
        pad_frames[0],
        vmin=0,
        vmax=adc_max,
        cmap="hot",
        interpolation="nearest",
        origin="upper",
    )
    heatmap_axes.set_title("Pad 10x10 pressure")
    heatmap_axes.set_xlabel("Sensor column")
    heatmap_axes.set_ylabel("Sensor row")
    figure.colorbar(heatmap, ax=heatmap_axes, label="ADC reading")

    plot_axes.plot(times, maximum_pressure, label="Maximum", linewidth=2)
    plot_axes.plot(times, mean_pressure, label="Mean", linewidth=2)
    cursor = plot_axes.axvline(times[0], color="black", linestyle="--", linewidth=1)
    plot_axes.set_xlim(times[0], times[-1])
    plot_axes.set_ylim(0, max(adc_max, int(maximum_pressure.max()) + 1))
    plot_axes.set_xlabel("Time (seconds)")
    plot_axes.set_ylabel("Pressure (ADC)")
    plot_axes.set_title("Pressure over time")
    plot_axes.grid(alpha=0.25)
    plot_axes.legend(loc="upper right")

    def update(frame):
        heatmap.set_data(pad_frames[frame])
        cursor.set_xdata([times[frame]])
        heatmap_axes.set_title(f"Pad pressure at {times[frame]:.2f} s")
        return heatmap, cursor

    heatmap_animation = animation.FuncAnimation(
        figure,
        update,
        frames=len(pad_frames),
        interval=1000 / float(fs),
        blit=False,
    )
    figure.tight_layout()

    backend = plt.get_backend().lower()

    if show:
        if backend == "agg": # If no drawing backend (e.g. im on WSL linux) then just save GIF
            output = Path(save_path or "pad_pressure.gif")
            output.parent.mkdir(parents=True, exist_ok=True)
            heatmap_animation.save(output, writer="pillow", fps=float(fs))
            print(f"Saved GIF animation to {output.resolve()}")
            plt.close(figure)
        else: # Use tkinter 'TkAgg' or 'QtAgg' if available
            plt.show()

    return heatmap_animation

 