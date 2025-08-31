# === Core Libraries ===
import numpy as np
import textwrap

# === Visualisation ===
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from IPython.display import display

# === Widgets for Interactivity ===
from ipywidgets import Dropdown, HBox, IntSlider, VBox, interactive_output

# === Image Processing ===
from skimage.exposure import adjust_gamma
from skimage.filters import threshold_local
from skimage.morphology import disk, white_tophat

# === Progress Bar ===
from tqdm.auto import tqdm


def plot_hist(ax, data, title=None):
    """
    Helper function to plot a histogram of intensity values on a given axis.
    """
    ax.hist(data.ravel(), bins=512, color='gray', density=True)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.set_xlim(0, np.max(data))
    ax.tick_params(axis='x', which='minor', length=5, color='gray')
    if title:
        ax.set_title("\n".join(textwrap.wrap(title, width=30)), fontsize=10)


def plot_channel_histograms(ioi_files, ioi_files_name=None):
    """
    Plots static histograms for each channel in each file.
    
    Parameters:
        ioi_files (dict): Dictionary of image stacks indexed by file.
        ioi_files_name (dict): Optional dictionary of file names.
    """
    n_files = len(ioi_files)
    example_file = next(iter(ioi_files.values()))
    if example_file.ndim == 4:
        n_channels = example_file.shape[3]
    elif example_file.ndim == 5:
        n_channels = example_file.shape[4]
    else:
        raise ValueError("Data must be 4D or 5D with channels last.")

    fig, axs = plt.subplots(n_files, n_channels, figsize=(3 * n_channels, 3 * n_files))
    axs = np.atleast_2d(axs)

    for file_idx, key in enumerate(ioi_files):
        data = ioi_files[key]
        fname = ioi_files_name[key] if ioi_files_name else str(key)
        for ch in range(n_channels):
            channel_data = data[..., ch]
            plot_hist(axs[file_idx, ch], channel_data, title=f"{fname}\nChannel {ch}")

    for i in range(n_files * n_channels, axs.size):
        fig.delaxes(axs.flat[i])

    plt.tight_layout()
    plt.show()


def interactive_histogram_viewer_channel(ioi_files, ioi_files_name=None):
    """
    Launches an interactive widget for browsing histograms over time and channels.
    
    Parameters:
        ioi_files (dict): Dictionary of 4D arrays [T, Y, X, C].
        ioi_files_name (dict): Optional dictionary of file names.
    """
    file_keys = list(ioi_files.keys())
    first_file = ioi_files[file_keys[0]]
    T, Y, X, C = first_file.shape

    file_dropdown = Dropdown(options=file_keys, description="File")
    t_slider = IntSlider(min=0, max=T - 1, description="Time (t)")
    ch_slider = IntSlider(min=0, max=C - 1, description="Channel")

    def update(file, t, ch):
        data = ioi_files[file]
        slice_data = data[t, :, :, ch]
        title = f"{file} - {ioi_files_name[file]}" if ioi_files_name else str(file)
        title += f"\nT={t}, Channel={ch}"

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_hist(ax, slice_data, title=title)
        plt.tight_layout()
        plt.show()

    controls = VBox([file_dropdown, HBox([t_slider, ch_slider])])
    output = interactive_output(update, {'file': file_dropdown, 't': t_slider, 'ch': ch_slider})
    display(controls, output)
    
    



def get_mixing_matrix(matrix="default"):
    """
    Returns a mixing matrix based on a preset name or a user-defined matrix.

    Parameters:
        matrix (str or np.ndarray): Either a string preset name or a 2D NumPy array.

    Returns:
        np.ndarray: The mixing matrix to be used for channel unmixing.
    """
    if isinstance(matrix, str):
        if matrix == "default":
            return np.array([
                [0.566, 0.035, 0.210],  # Ch0: ET620/60m
                [0.000, 0.697, 0.001],  # Ch1: ET525/50m
                [0.091, 0.000, 0.464],  # Ch2: ET700/75m
            ])
        else:
            raise ValueError(f"Unknown mixing matrix preset: '{matrix}'")

    elif isinstance(matrix, np.ndarray):
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            return matrix
        else:
            raise ValueError("Custom mixing matrix must be a square 2D numpy array.")

    else:
        raise TypeError("mixing_matrix must be a string preset or a numpy.ndarray.")


def unmix_channels_by_frame(channels, mixing_matrix):
    """
    Unmix fluorescence channels frame-by-frame using the pseudo-inverse of a mixing matrix.
    
    Args:
        channels (np.ndarray): 4D stack with shape (T, H, W, C).
        mixing_matrix (np.ndarray): Crosstalk matrix (C x C).
    
    Returns:
        np.ndarray: Unmixed stack of shape (T, H, W, C).
    """
    T, H, W, C = channels.shape
    inv_M = np.linalg.pinv(mixing_matrix)
    unmixed = np.zeros_like(channels, dtype=np.float32)

    for t in range(T):
        flat = channels[t].reshape(-1, C)
        unmixed_flat = np.dot(flat, inv_M.T)
        unmixed[t] = np.clip(unmixed_flat.reshape(H, W, C), 0, None)

    return unmixed


def background_correction(image_stack, radius=20):
    """
    Apply white tophat background correction.
    
    Args:
        image_stack (np.ndarray): 4D stack (T, H, W, C)
        radius (int): Radius of the structuring element.
    
    Returns:
        np.ndarray: Background-corrected image stack.
    """
    selem = disk(radius)
    frames, height, width, channels = image_stack.shape
    corrected_stack = np.empty_like(image_stack)

    for c in tqdm(range(channels), desc="BG Correction - Channels", leave=False):
        for t in range(frames):
            corrected_stack[t, :, :, c] = white_tophat(image_stack[t, :, :, c], selem)

    return corrected_stack


def apply_local_threshold_subtract(im, block_size=35, offset=0):
    """
    Local threshold subtraction using skimage's threshold_local.

    Args:
        im (np.ndarray): Input image stack (T, H, W, C)
        block_size (int): Size of neighbourhood.
        offset (float): Offset from mean.

    Returns:
        np.ndarray: Threshold-subtracted image stack.
    """
    T, H, W, C = im.shape
    im_thresh = np.zeros_like(im)

    for t in range(T):
        for c in range(C):
            local_thresh = threshold_local(im[t, :, :, c], block_size=block_size, offset=offset)
            im_thresh[t, :, :, c] = np.clip(im[t, :, :, c] - local_thresh, a_min=0, a_max=None)

    return im_thresh


def pre_processing(
    ioi_files,
    time_interval=None,
    channels_to_use=3,
    processing_steps=None,  # NEW
    mixing_matrix="default",
    low_percentiles=[1, 0, 0.1],
    high_percentiles=[99.9, 99.9, 99.9],
    gamma=None
):
    """
    Flexible pre-processing pipeline for time-lapse fluorescence microscopy data.

    Parameters:
        ioi_files (dict): Raw 4D stacks (T, H, W, C) per file.
        time_interval (int or None): Trim time dimension to this length.
        channels_to_use (int): Number of channels to retain.
        processing_steps (list): Ordered list of steps. Allowed: 
            ["trim", "background", "threshold", "unmix", "normalize", "gamma"]
        mixing_matrix (str or np.ndarray): Crosstalk matrix or preset name.
        low_percentiles (list): Per-channel lower clip values.
        high_percentiles (list): Per-channel upper clip values.
        gamma (float or None): If set, apply gamma correction.

    Returns:
        dict: Processed stacks.
    """
    if processing_steps is None:
        processing_steps = ["trim", "unmix", "normalize"]

    ioi_processing = {}
    mixing_matrix = get_mixing_matrix(mixing_matrix)

    for key, im_seq in tqdm(ioi_files.items(), desc="Processing files"):
        im = im_seq[..., :channels_to_use]

        for step in processing_steps:
            if step == "trim" and time_interval:
                im = im[:time_interval]

            elif step == "background":
                im = background_correction(im)

            elif step == "threshold":
                im = apply_local_threshold_subtract(im)

            elif step == "unmix":
                im = unmix_channels_by_frame(im, mixing_matrix)

        # Always create a float32 output
        processed_im = np.empty_like(im, dtype=np.float32)
        n_channels = im.shape[3]

        for ch in range(n_channels):
            ch_im = im[..., ch]
            norm = ch_im.astype(np.float32)

            if "normalize" in processing_steps:
                low, high = np.percentile(ch_im, (low_percentiles[ch], high_percentiles[ch]))
                norm = np.clip((ch_im - low) / (high - low), 0, 1)

            if "gamma" in processing_steps and gamma is not None:
                norm = adjust_gamma(norm, gamma)

            processed_im[..., ch] = norm

        ioi_processing[key] = processed_im

    return ioi_processing