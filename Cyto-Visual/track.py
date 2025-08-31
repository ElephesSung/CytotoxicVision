import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from btrack.btypes import PyTrackObject
import btrack

def run_btrack(
    ioi_region,
    ioi_masks,
    feature_dictionary,
    ioi_processing=None,
    ioi_files=None,
    config_path=None,
    max_search_radius=55,
    optimizer_time_limit=int(12e4),
):
    """
    Perform Bayesian tracking on each file/channel using `btrack`.

    Parameters:
        ioi_region (dict): Region properties per file and channel.
        ioi_masks (dict): Corresponding mask stack to extract image dimensions.
        feature_dictionary (dict): Dictionary specifying which features to use for each label.
        ioi_processing (dict or None): Pre-processed image stack (for intensity features).
        ioi_files (dict or None): Raw image stack (for raw intensity features).
        config_path (str): Path to the btrack JSON configuration file.
        max_search_radius (int): Maximum movement radius between frames.
        optimizer_time_limit (int): Time budget for optimisation step.

    Returns:
        ioi_tracking (dict): Per-file and per-channel tracking dataframes.
        ioi_napari_tracks (dict): Napari-friendly array [track_id, t, y, x].
    """
    ioi_tracking = {}
    ioi_napari_tracks = {}

    for file_seq in tqdm(ioi_region, desc='Tracking files'):
        df_file = ioi_region[file_seq]
        ioi_tracking[file_seq] = {}
        ioi_napari_tracks[file_seq] = {}

        for label in tqdm(df_file, desc='Tracking channels', leave=False):
            df_file_sub = df_file[label]
            objects = []

            for _, row in df_file_sub.iterrows():
                obj = PyTrackObject()
                obj.t, obj.x, obj.y, obj.z = row['frame'], row['x'], row['y'], 0
                obj.properties = {
                    "area": row.get("area", np.nan),
                    'coords': json.dumps(row["coords"].tolist()) if "coords" in row else None
                }

                # Add optional features
                for ch in [0, 1, 2]:
                    for prefix in ["intensity_mean", "raw_intensity_mean"]:
                        key = f"{prefix}_ch{ch}"
                        if key in row:
                            obj.properties[key] = row[key]

                objects.append(obj)

            # Run Bayesian Tracker
            T, H, W = ioi_masks[file_seq].shape[:3]
            features = feature_dictionary.get(label, [])

            with btrack.BayesianTracker() as tracker:
                if config_path:
                    tracker.configure(config_path)
                else:
                    raise ValueError("Must specify a valid btrack configuration path.")

                tracker.max_search_radius = max_search_radius
                tracker.tracking_updates = ["MOTION", "VISUAL"]
                tracker.features = features
                tracker.append(objects)
                tracker.volume = ((0, W), (0, H))
                tracker.track(step_size=1)
                tracker.optimize(optimizer_options={'tm_lim': optimizer_time_limit})

                # Convert tracking results to dataframe
                tracks = tracker.tracks
                track_data = []

                for track in tracks:
                    tdic = track.to_dict()
                    for i in range(len(tdic['t'])):
                        coords_raw = tdic['coords'][i]
                        try:
                            coords = np.array(json.loads(coords_raw)) if coords_raw else np.array([])
                        except json.JSONDecodeError:
                            coords = np.array([])

                        entry = {
                            'id': tdic['ID'],
                            't': tdic['t'][i],
                            'y': tdic['y'][i],
                            'x': tdic['x'][i],
                            'coords': coords,
                            'dummy': tdic['dummy'][i],
                            'area': tdic.get('area', [np.nan]*len(tdic['t']))[i],
                        }

                        for ch in [0, 1, 2]:
                            for prefix in ["intensity_mean", "raw_intensity_mean"]:
                                key = f"{prefix}_ch{ch}"
                                entry[key] = tdic.get(key, [np.nan]*len(tdic['t']))[i]

                        track_data.append(entry)

                df_t = pd.DataFrame(track_data)
                ioi_tracking[file_seq][label] = df_t
                ioi_napari_tracks[file_seq][label] = df_t[['id', 't', 'y', 'x']].to_numpy()

    return ioi_tracking, ioi_napari_tracks