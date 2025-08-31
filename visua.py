import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from IPython.display import HTML
from tqdm.auto import tqdm
import os

def merge_view(image1, image2, image3, brightfield,
               blue_brightness_factor=1.0, 
               green_brightness_factor=1.0, 
               red_brightness_factor=1.0, 
               brightfield_intensity_factor=1.0):
    def normalize_and_scale(image, brightness_factor):
        normalized = image / np.max(image) if np.max(image) > 0 else image
        return np.clip(normalized * brightness_factor, 0, 1)
    
    blue_channel = normalize_and_scale(image1, blue_brightness_factor)
    green_channel = normalize_and_scale(image2, green_brightness_factor)
    red_channel = normalize_and_scale(image3, red_brightness_factor)
    brightfield_channel = normalize_and_scale(brightfield, brightfield_intensity_factor)

    merged = np.zeros((*image1.shape, 3), dtype=np.float32)
    merged[..., 0] = np.clip(red_channel + brightfield_channel, 0, 1)
    merged[..., 1] = np.clip(green_channel + brightfield_channel, 0, 1)
    merged[..., 2] = np.clip(blue_channel + brightfield_channel, 0, 1)
    return merged


def mat_visua(ioi_files, size_para=4, cmap_list=None, interval=300,
                                   save=False, filename="animation", format="gif"):
    """
    Displays and optionally saves animated multichannel microscopy data.

    Parameters:
        ioi_files (dict): Dictionary of indexed 4D arrays.
        size_para (int): Figure size multiplier.
        cmap_list (list): List of colormaps for channels.
        interval (int): Time between frames in ms.
        save (bool): Whether to save the animation.
        filename (str): Name of output file (without extension).
        format (str): 'gif' or 'mp4'.

    Returns:
        HTML: Animation display for Jupyter.
    """
    if cmap_list is None:
        cmap_list = ['Blues', 'Greens', 'Reds', 'gray']

    num_files = len(ioi_files)
    Channel_No = next(iter(ioi_files.values())).shape[3]

    fig, axes = plt.subplots(num_files, Channel_No + 1,
                             figsize=((Channel_No + 1) * size_para, num_files * size_para),
                             sharey=True)
    if num_files == 1:
        axes = np.expand_dims(axes, axis=0)
    if Channel_No + 1 == 1:
        axes = np.expand_dims(axes, axis=1)

    for t in range(Channel_No + 1):
        title = f"Channel {t+1}" if t < Channel_No else "Merged View"
        axes[0, t].set_title(title, fontsize=14, fontweight='bold')

    for c in range(num_files):
        row_mid_y = (axes[c, 0].get_position().y0 + axes[c, 0].get_position().y1) / 2
        fig.text(0.06, row_mid_y, f"File {c+1}", fontsize=14, fontweight='bold',
                 ha='center', va='center', rotation=90)

    ioi_files_SC = {}
    images_displayed = []

    for idx, image in tqdm(ioi_files.items(), desc="Preparing frames"):
        ioi_files_SC[idx] = {}
        row_images = []
        for C in range(Channel_No):
            ioi_files_SC[idx][C + 1] = image[:, :, :, C]
            im = axes[idx, C].imshow(image[0, :, :, C], animated=True, cmap=cmap_list[C])
            axes[idx, C].grid(False)
            row_images.append(im)

        merged_image = merge_view(
            ioi_files_SC[idx][1][0],
            ioi_files_SC[idx][2][0],
            ioi_files_SC[idx][3][0],
            ioi_files_SC[idx][4][0]
        )
        im_merged = axes[idx, Channel_No].imshow(merged_image, animated=True)
        axes[idx, Channel_No].grid(False)
        row_images.append(im_merged)
        images_displayed.append(row_images)

    frame_texts = []
    for ax_row in axes:
        row_texts = []
        for j, ax in enumerate(ax_row):
            color = "black" if j < Channel_No - 1 else "white"
            text = ax.text(0.95, 0.05, "", transform=ax.transAxes, fontsize=8,
                           color=color, ha="right", va="bottom", animated=True)
            row_texts.append(text)
        frame_texts.append(row_texts)

    def update_all(frame_idx):
        updated_elements = []
        for idx, row_images in enumerate(images_displayed):
            for C, im in enumerate(row_images):
                if C < Channel_No:
                    im.set_array(ioi_files_SC[idx][C + 1][frame_idx])
                else:
                    merged_image = merge_view(
                        ioi_files_SC[idx][1][frame_idx],
                        ioi_files_SC[idx][2][frame_idx],
                        ioi_files_SC[idx][3][frame_idx],
                        ioi_files_SC[idx][4][frame_idx]
                    )
                    im.set_array(merged_image)
                frame_texts[idx][C].set_text(f"Frame {frame_idx}")
                updated_elements.append(im)
                updated_elements.append(frame_texts[idx][C])
        return updated_elements

    num_frames = next(iter(ioi_files.values())).shape[0]
    anim = FuncAnimation(fig, update_all, frames=num_frames, interval=interval, blit=True)

    if save:
        ext = format.lower()
        filepath = f"{filename}.{ext}"
        if ext == "gif":
            writer = PillowWriter(fps=1000 // interval)
        elif ext == "mp4":
            writer = FFMpegWriter(fps=1000 // interval)
        else:
            raise ValueError("Format must be 'gif' or 'mp4'")

        print(f"Saving animation to {filepath} ...")
        anim.save(filepath, writer=writer)
        print("Done.")

    plt.close(fig)
    return HTML(anim.to_jshtml())