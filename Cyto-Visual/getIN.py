# === Standard Library ===
import os
from glob import glob

# === Third-Party Libraries ===
import numpy as np
from tqdm.auto import tqdm  # Auto-detects notebook vs. terminal

# === Image I/O and Handling ===
import pims
from czifile import CziFile
from nd2reader import ND2Reader

def load_tiff_sequence(folder_path, verbose=True):
    """
    Loads a sequence of .tif images from the specified folder using pims.

    Parameters:
        folder_path (str): Path to the folder containing .tif files.
        verbose (bool): Whether to print progress and summary. Default is True.

    Returns:
        image_dict (dict): Dictionary of images indexed by integer keys.
        name_dict (dict): Dictionary of file names indexed by integer keys.
    """
    # Find and sort .tif files
    tif_files = sorted(glob(os.path.join(folder_path, "*.tif")))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {folder_path}")

    # Load the sequence
    images = pims.ImageSequence(tif_files)

    # Initialise dictionaries
    image_dict = {}
    name_dict = {}

    for idx, (frame, file_name) in tqdm(enumerate(zip(images, tif_files)), 
                                        desc="Loading .tif files", 
                                        total=len(images)):
        image_dict[idx] = frame
        name_dict[idx] = os.path.basename(file_name)
        if verbose:
            print(f"Loaded frame {idx}: Shape = {frame.shape}, File = {name_dict[idx]}")

    if verbose:
        print(f"Imported {len(images)} .tif files from {folder_path}")

    return image_dict, name_dict




def load_czi_sequence(folder_path, verbose=True):
    """
    Loads a sequence of .czi files from the specified folder using czifile.

    Parameters:
        folder_path (str): Path to the folder containing .czi files.
        verbose (bool): Whether to print progress and summary.

    Returns:
        image_dict (dict): Dictionary of images indexed by integer keys.
        name_dict (dict): Dictionary of file names indexed by integer keys.
    """
    czi_files = sorted(glob(os.path.join(folder_path, "*.czi")))
    if not czi_files:
        raise FileNotFoundError(f"No .czi files found in {folder_path}")

    image_dict = {}
    name_dict = {}

    for idx, file_path in tqdm(enumerate(czi_files), desc="Loading .czi files", total=len(czi_files)):
        with CziFile(file_path) as czi:
            image = czi.asarray()
            image = np.squeeze(image)  # remove singleton dimensions if needed
            image_dict[idx] = image
            name_dict[idx] = os.path.basename(file_path)

            if verbose:
                print(f"Loaded frame {idx}: Shape = {image.shape}, File = {name_dict[idx]}")

    if verbose:
        print(f"Imported {len(czi_files)} .czi files from {folder_path}")

    return image_dict, name_dict



def load_nd2_sequence(folder_path, verbose=True):
    """
    Loads a sequence of .nd2 files from the specified folder using nd2reader.

    Parameters:
        folder_path (str): Path to the folder containing .nd2 files.
        verbose (bool): Whether to print progress and summary.

    Returns:
        image_dict (dict): Dictionary of images indexed by integer keys.
        name_dict (dict): Dictionary of file names indexed by integer keys.
    """
    nd2_files = sorted(glob(os.path.join(folder_path, "*.nd2")))
    if not nd2_files:
        raise FileNotFoundError(f"No .nd2 files found in {folder_path}")

    image_dict = {}
    name_dict = {}

    for idx, file_path in tqdm(enumerate(nd2_files), desc="Loading .nd2 files", total=len(nd2_files)):
        with ND2Reader(file_path) as images:
            images.bundle_axes = 'yx'  # Load frames as 2D
            frame = np.array(images[0])  # Load first frame only
            image_dict[idx] = frame
            name_dict[idx] = os.path.basename(file_path)

            if verbose:
                print(f"Loaded frame {idx}: Shape = {frame.shape}, File = {name_dict[idx]}")

    if verbose:
        print(f"Imported {len(nd2_files)} .nd2 files from {folder_path}")

    return image_dict, name_dict