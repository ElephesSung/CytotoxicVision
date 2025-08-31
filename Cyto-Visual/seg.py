import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from skimage.measure import label, regionprops
from cellpose import models
from celldetective.segmentation import merge_instance_segmentation
from tqdm.auto import tqdm

mpl.rcParams['animation.embed_limit'] = 1000
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "legend.fontsize": 14,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "savefig.format": "pdf",
    "legend.edgecolor": "0.0",
    "legend.framealpha": 1.0,
})

def run_cellpose_segmentation(
    processed_data,
    model_type='cyto3',
    flow_thresholds=[0.4, 0.4, 0.6],
    cellprob_threshold=None,
    diameter=None,
    channels=[0, 0],
    augment=False,
    resample=True,
    do_3D=False,
    gpu=False,
    model_instance=None
):
    """
    Apply Cellpose segmentation to each channel and frame in pre-processed image stacks.

    Parameters:
        processed_data (dict): Dictionary of image stacks [T, Y, X, C].
        model_type (str): Cellpose model type ('cyto', 'cyto2', 'cyto3', 'nuclei').
        flow_thresholds (list of float): Flow threshold per channel.
        cellprob_threshold (float or None): Optional cell probability threshold.
        diameter (float or None): Estimated cell diameter (in pixels).
        channels (list): List of [cytoplasm_channel, nucleus_channel] — typically [0, 0] for grayscale.
        augment (bool): Whether to run augmentation (slower, more accurate).
        resample (bool): Whether to resize images to diameter.
        do_3D (bool): Use 3D segmentation (if data is 3D).
        gpu (bool): Use GPU acceleration.
        model_instance (Cellpose or None): Pass a pre-initialised Cellpose model.

    Returns:
        dict: Segmentation masks with shape [T, Y, X, C] per file.
    """
    # Initialise model
    model = model_instance or models.Cellpose(gpu=gpu, model_type=model_type)
    segmentation_masks = {}

    for file_key in tqdm(processed_data, desc='Files'):
        image_stack = processed_data[file_key]
        T, H, W, C = image_stack.shape
        masks_per_channel = []

        for ch in tqdm(range(C), desc='Channels', leave=False):
            channel_stack = image_stack[..., ch]

            channel_masks = []
            for t, frame in enumerate(tqdm(channel_stack, desc='Frames', leave=False)):
                mask, *_ = model.eval(
                    frame,
                    diameter=diameter,
                    channels=channels,
                    flow_threshold=flow_thresholds[ch] if isinstance(flow_thresholds, list) else flow_thresholds,
                    cellprob_threshold=cellprob_threshold,
                    augment=augment,
                    resample=resample,
                    do_3D=do_3D
                )
                channel_masks.append(mask)

            masks_per_channel.append(np.array(channel_masks))

        segmentation_masks[file_key] = np.transpose(np.array(masks_per_channel), (1, 2, 3, 0))

    return segmentation_masks


def plot_mask_area_distributions(
    ioi_masks,
    file_names=None,
    channel_labels=None,
    bins=50,
    xlim=None,
    save_path=None  
):
    """
    Plot overlaid mask area histograms for each file and channel.

    Parameters:
        ioi_masks (dict): Dictionary of 4D arrays (T, H, W, C) with mask labels or binary masks.
        file_names (dict or None): Optional mapping from file index to filename.
        channel_labels (list or None): Labels for each channel. Default: ["Channel 0", "Channel 1", ...]
        bins (int): Number of bins in the histogram (default: 50).
        xlim (tuple or None): (xmin, xmax) to set the x-axis range. Default: auto.
        save_path (str or None): Path to save the figure (e.g., "output.svg" or "figure.pdf"). If None, does not save.
    """
    # --- Extract area data ---
    all_area_distributions = {}

    for file_seq in tqdm(ioi_masks, desc='Collecting areas', total=len(ioi_masks)):
        mask_file = ioi_masks[file_seq]
        area_dict = {}
        for ch in range(mask_file.shape[3]):
            areas = []
            for t in range(mask_file.shape[0]):
                mask_channel = mask_file[t, :, :, ch]
                labelled = label(mask_channel) if mask_channel.max() <= 1 else mask_channel
                props = regionprops(labelled)
                areas.extend([p.area for p in props])
            area_dict[ch] = areas
        all_area_distributions[file_seq] = area_dict

    # --- Setup plot ---
    num_files = len(all_area_distributions)
    fig, axes = plt.subplots(num_files, 1, figsize=(6, 4 * num_files), squeeze=False, dpi=250)

    n_channels = ioi_masks[next(iter(ioi_masks))].shape[3]
    default_labels = [f"Channel {i}" for i in range(n_channels)]
    default_colors = ["dodgerblue", "tomato", "seagreen", "gold", "mediumpurple"]

    channel_labels = channel_labels or default_labels

    for i, (file_seq, area_dict) in enumerate(all_area_distributions.items()):
        ax = axes[i, 0]
        for ch in range(n_channels):
            areas = area_dict.get(ch, [])
            ax.hist(
                areas,
                bins=bins,
                density=True,
                alpha=0.5,
                color=default_colors[ch % len(default_colors)],
                label=channel_labels[ch] if ch < len(channel_labels) else f"Channel {ch}",
                edgecolor='none'
            )

        # File title
        file_title = (
            file_names[file_seq] if file_names and file_seq in file_names
            else f"File {file_seq}"
        )
        ax.set_title(f"File {file_seq}: {file_title}", fontsize=14, fontweight='bold')

        ax.set_xlabel('Mask Area (pixels)')
        ax.set_ylabel('Density')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        if xlim is not None:
            ax.set_xlim(*xlim)

    plt.tight_layout()

    # --- Save or show ---
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()
        


def clean_masks(ioi_masks, area_threshold=40, channels=None):
    """
    Remove small labelled masks based on area threshold.

    Parameters:
        ioi_masks (dict): Dictionary of 4D arrays (T, H, W, C) containing labelled masks.
        area_threshold (int): Minimum area for a mask to be kept.
        channels (list or None): Channels to process. If None, all channels will be processed.

    Returns:
        dict: Dictionary with small masks removed.
    """
    cleaned_ioi_masks = {}

    for file_seq, mask_file in ioi_masks.items():
        T, H, W, C = mask_file.shape
        cleaned = mask_file.copy()

        # Set default channels if not provided
        current_channels = channels if channels is not None else list(range(C))

        for ch in current_channels:
            for t in range(T):
                frame = mask_file[t, :, :, ch]
                for prop in regionprops(frame):
                    if prop.area < area_threshold:
                        cleaned[t, :, :, ch][frame == prop.label] = 0

        cleaned_ioi_masks[file_seq] = cleaned

    return cleaned_ioi_masks

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from skimage.measure import label, regionprops
from celldetective.segmentation import merge_instance_segmentation

def region_info(
    ioi_masks,
    ioi_processing,
    ioi_files,
    merge_channels=(1, 2),
    merge_mode='OR',
    iou_threshold=0.04,
    extract_channels=(0, 1, 2),
    region_features=("area", "coords", "centroid", "label"),
    include_intensity=True,
    include_raw_intensity=True,
    time_trim=None
):
    """
    Merge instance masks and extract region statistics across time-lapse image stacks.

    Parameters:
        ioi_masks (dict): Mask data (T, H, W, C).
        ioi_processing (dict): Preprocessed image stacks (T, H, W, C).
        ioi_files (dict): Raw image stacks (T, H, W, C).
        merge_channels (tuple): Channels to merge, e.g. (1, 2).
        merge_mode (str): Merge mode for instance segmentation.
        iou_threshold (float): IOU threshold for merging instances.
        extract_channels (tuple): Channels to extract intensity info from.
        region_features (tuple): Regionprops features to extract (area, coords, centroid, label).
        include_intensity (bool): Extract processed image intensities.
        include_raw_intensity (bool): Extract raw image intensities.
        time_trim (int or None): If set, only process the first N time frames.

    Returns:
        ioi_region (dict): Per-file region dataframes.
        ioi_masks_merge (dict): Per-file merged masks (T, H, W, 2).
    """
    ioi_region = {}
    ioi_masks_merge = {}

    for image_seq in tqdm(ioi_masks, desc='Files'):
        mask_file = ioi_masks[image_seq]
        T, H, W, C = mask_file.shape
        t_max = time_trim if time_trim is not None else T

        # --- Channel 0 (not merged) ---
        mask_ch0 = mask_file[:t_max, :, :, 0]

        # --- Merged mask (frame by frame) ---
        mask_merge = []
        for t in range(t_max):
            ch_frames = [mask_file[t, :, :, ch] for ch in merge_channels]
            merged = merge_instance_segmentation(
                ch_frames,
                mode=merge_mode,
                iou_matching_threshold=iou_threshold
            )
            mask_merge.append(merged)
        mask_merge = np.stack(mask_merge, axis=0)

        ioi_masks_merge[image_seq] = np.stack([mask_ch0, mask_merge], axis=-1)

        mask_ch0_region = []
        mask_merge_region = []

        for mask_stack, region_list in zip([mask_ch0, mask_merge], [mask_ch0_region, mask_merge_region]):
            for t in range(t_max):
                regions = regionprops(mask_stack[t])
                for region in regions:
                    props = {'frame': t, 'label': region.label}

                    if "area" in region_features:
                        props['area'] = region.area
                    if "coords" in region_features:
                        props['coords'] = region.coords
                    if "centroid" in region_features:
                        props['y'], props['x'] = region.centroid
                        props['z'] = 0

                    if include_intensity:
                        for ch in extract_channels:
                            img = ioi_processing[image_seq][t, :, :, ch]
                            props[f'intensity_mean_ch{ch}'] = np.mean(
                                img[region.coords[:, 0], region.coords[:, 1]]
                            )

                    if include_raw_intensity:
                        for ch in extract_channels:
                            raw = ioi_files[image_seq][t, :, :, ch]
                            props[f'raw_intensity_mean_ch{ch}'] = np.mean(
                                raw[region.coords[:, 0], region.coords[:, 1]]
                            )

                    region_list.append(props)

        # --- Determine merged key name ---
        if isinstance(merge_channels, (tuple, list)) and len(merge_channels) == 2:
            merge_key = f'ch{merge_channels[0]}{merge_channels[1]}'
        else:
            merge_key = 'merged'

        ioi_region[image_seq] = {
            'ch0': pd.DataFrame(mask_ch0_region),
            merge_key: pd.DataFrame(mask_merge_region)
        }

    return ioi_region, ioi_masks_merge