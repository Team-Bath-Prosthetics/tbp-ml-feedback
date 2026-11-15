# viz

Plotting and animation helpers for tactile recordings.

## Showing plots in a window

`plt.show()` needs a GUI toolkit. If you get

```
UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
```

then matplotlib couldn't find one and fell back to `Agg`, which only renders to files. Install Tk for the Python your venv was built from (no reinstall of the venv needed):

| Platform | Fix |
|---|---|
| Linux (Ubuntu/Debian/Pop!_OS) | `sudo apt install python3-tk` |
| Mac, Homebrew Python | `brew install python-tk@3.11` (match your Python version) |
| Mac/Windows, python.org installer | Included by default (Windows: the "tcl/tk and IDLE" option) |
| conda | Included |

In tests, use `matplotlib.use("Agg")` and don't call `plt.show()`. Tests shouldn't open windows.
