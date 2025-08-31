# --- Standard Library ---
import textwrap

# --- Numerical & Data Handling ---
import numpy as np
import pandas as pd

# --- Plotting & Visualisation ---
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML

# --- Machine Learning ---
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- Utilities ---
from tqdm.auto import tqdm

def plot_joint_fluo(
    df,
    file_name="",
    mode="log",  # "log", "linear", "norm", "lognorm", or "minmax"
    x_col="raw_intensity_mean_ch1",
    y_col="raw_intensity_mean_ch2",
    bins=60,
    height=6,
    alpha=0.6,
    color="#7593af"
):
    """
    mode:
        "log"      - log10 scale
        "linear"   - raw values
        "norm"     - z-score normalization (mean=0, std=1)
        "lognorm"  - log10 then z-score normalization
        "minmax"   - scale each axis to [0, 1] independently
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    if mode == "log":
        x = np.log10(df[x_col].replace(0, np.nan))
        y = np.log10(df[y_col].replace(0, np.nan))
        xlabel = r"$\log_{10}(\mathrm{Calcium\ Green})$"
        ylabel = r"$\log_{10}(\mathrm{TO\text{-}PRO\text{-}3})$"
    elif mode == "linear":
        x = df[x_col]
        y = df[y_col]
        xlabel = "Calcium Green intensity"
        ylabel = "TO-PRO-3 intensity"
    elif mode == "norm":
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x = pd.Series(scaler_x.fit_transform(df[[x_col]].replace(0, np.nan)).ravel(), index=df.index)
        y = pd.Series(scaler_y.fit_transform(df[[y_col]].replace(0, np.nan)).ravel(), index=df.index)
        xlabel = "Normalized Calcium Green"
        ylabel = "Normalized TO-PRO-3"
    elif mode == "lognorm":
        # Log10, then z-score normalization
        x_log = np.log10(df[x_col].replace(0, np.nan))
        y_log = np.log10(df[y_col].replace(0, np.nan))
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x = pd.Series(scaler_x.fit_transform(x_log.values.reshape(-1, 1)).ravel(), index=df.index)
        y = pd.Series(scaler_y.fit_transform(y_log.values.reshape(-1, 1)).ravel(), index=df.index)
        xlabel = "Log-Normalized Calcium Green"
        ylabel = "Log-Normalized TO-PRO-3"
    elif mode == "minmax":
        # Scale to [0, 1] independently
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        x = pd.Series(scaler_x.fit_transform(df[[x_col]].replace(0, np.nan)).ravel(), index=df.index)
        y = pd.Series(scaler_y.fit_transform(df[[y_col]].replace(0, np.nan)).ravel(), index=df.index)
        xlabel = "MinMax Calcium Green"
        ylabel = "MinMax TO-PRO-3"
    else:
        raise ValueError("mode must be 'log', 'linear', 'norm', 'lognorm', or 'minmax'")

    mask = ~(x.isna() | y.isna())
    plot_df = df.loc[mask].copy()
    plot_df["x_plot"] = x[mask]
    plot_df["y_plot"] = y[mask]

    g = sns.jointplot(
        data=plot_df,
        x="x_plot",
        y="y_plot",
        kind="scatter",
        marginal_kws=dict(bins=bins, fill=True, color=color, alpha=alpha),
        height=height,
        space=0.1,
        s=10,
        joint_kws=dict(s=14, edgecolor=None, alpha=0.3, color=color)
    )

    g.set_axis_labels(xlabel, ylabel)
    g.fig.set_dpi(150)
    g.fig.suptitle(f"{file_name}", y=1.02)

    for spine in g.ax_joint.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)
    for spine in g.ax_marg_x.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)
    for spine in g.ax_marg_y.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.show()

def classify_cell_states(
    ioi_tracking,
    ioi_files_name=None,
    use_gmm=False,
    use_log=True,
    use_raw=True,
    cluster_labels=("Live", "Dead"),
    dpi=250,
    figsize_scale=5,
    palette=None,
    point_size=15,
    xlim=None,
    ylim=None,
    n_clusters=2  # <--- add this parameter
):
    if palette is None:
        palette = {0: "seagreen", 1: "salmon"}

    num_files = len(ioi_tracking)
    fig, axes = plt.subplots(1, num_files, figsize=(num_files * figsize_scale, 5), sharex=False, sharey=False, dpi=dpi)
    if num_files == 1:
        axes = np.array([axes])

    for c in range(num_files):
        col_mid_x = (axes[c].get_position().x0 + axes[c].get_position().x1) / 2
        title = f"{ioi_files_name[c]}" if ioi_files_name else f"File {c+1}"
        fig.text(col_mid_x, 1.1, textwrap.fill(title, width=30), fontsize=12, fontweight='bold', ha='center')

    for c, seq in enumerate(ioi_tracking):
        df = ioi_tracking[seq]['ch12'].copy()
        x_key = "raw_intensity_mean_ch1" if use_raw else "intensity_mean_ch1"
        y_key = "raw_intensity_mean_ch2" if use_raw else "intensity_mean_ch2"

        feat = df[[x_key, y_key]].dropna()
        idx = feat.index

        X_cluster = np.log1p(feat.values) if use_log else feat.values
        scaler = RobustScaler()
        X_cluster_scaled = scaler.fit_transform(X_cluster)
        X_plot = np.log1p(feat.values) if use_log else feat.values

        # Use user-defined cluster number
        if use_gmm:
            model = GaussianMixture(n_components=n_clusters, covariance_type='full', n_init=30, random_state=0)
            labels = model.fit_predict(X_cluster_scaled)
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            labels = model.fit_predict(X_cluster_scaled)

        sil = silhouette_score(X_cluster_scaled, labels)
        print(f"File {c+1} silhouette score: {sil:.3f}")

        # Remap labels so that cluster 0 is always the one with the highest mean x (Calcium)
        centroids = [X_plot[labels == i].mean(axis=0) for i in range(n_clusters)]
        order = np.argsort([cent[0] for cent in centroids])[::-1]
        remapped_labels = np.zeros_like(labels)
        for new_label, old_label in enumerate(order):
            remapped_labels[labels == old_label] = new_label

        df.loc[idx, "Cell_Status"] = remapped_labels
        # Use cluster_labels if enough provided, else fallback to numbers
        if cluster_labels and len(cluster_labels) >= n_clusters:
            df.loc[idx, "Cell_Status_Label"] = [cluster_labels[int(i)] for i in remapped_labels]
        else:
            df.loc[idx, "Cell_Status_Label"] = [f"Cluster {int(i)}" for i in remapped_labels]
        ioi_tracking[seq]['ch12'] = df

        ax = axes[c]
        sns.scatterplot(
            x=X_plot[:, 0], y=X_plot[:, 1], hue=remapped_labels,
            palette=palette if n_clusters == 2 else None, ax=ax, s=point_size, alpha=0.6,
            edgecolor=None, legend=False
        )

        if use_gmm:
            xx, yy = np.meshgrid(
                np.linspace(X_plot[:, 0].min(), X_plot[:, 0].max(), 200),
                np.linspace(X_plot[:, 1].min(), X_plot[:, 1].max(), 200)
            )
            grid = np.column_stack([xx.ravel(), yy.ravel()])
            grid_scaled = scaler.transform(np.log1p(grid) if use_log else grid)
            Z = model.predict(grid_scaled).reshape(xx.shape)
            ax.contour(xx, yy, Z, levels=np.arange(n_clusters+1)-0.5, linewidths=1.2, colors='k', alpha=0.5)

        ax.set_xlim(xlim if xlim else (X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5))
        ax.set_ylim(ylim if ylim else (X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5))

        xlabel = "log-Calcium Green (raw)" if use_log else "Calcium Green (raw)"
        ylabel = "log-TO-PRO-3 (raw)" if use_log else "TO-PRO-3 (raw)"
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

        ax.set_facecolor('#e0f0ff')
        ax.grid(True, color='white', linestyle='-', linewidth=1.2)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Legend
        if cluster_labels and len(cluster_labels) >= n_clusters:
            handles = [
                plt.Line2D([], [], marker='o', linestyle='', color=palette[i] if palette and i in palette else f"C{i}")
                for i in range(n_clusters)
            ]
            ax.legend(handles, cluster_labels[:n_clusters], title="Cell Type", loc="upper right", fontsize=8)
        else:
            handles = [
                plt.Line2D([], [], marker='o', linestyle='', color=f"C{i}")
                for i in range(n_clusters)
            ]
            ax.legend(handles, [f"Cluster {i}" for i in range(n_clusters)], title="Cell Type", loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()

    return ioi_tracking

def threshold_cell_states(
    ioi_tracking,
    ioi_files_name=None,
    threshold_method="gmm",  # "gmm", "kde", "otsu", "yen", "li"
    use_log=True,
    use_raw=True,
    cluster_labels=("Alive", "Dead"),
    dpi=250,
    figsize_scale=5,
    palette=None,
    point_size=15,
    xlim=None,
    ylim=None,
    group_to_cluster={(1,0): 0, (0,0): 1, (0,1): 1, (1,1): 1},  # default: only (1,0) is alive
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import textwrap
    from sklearn.mixture import GaussianMixture
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks
    from skimage.filters import threshold_otsu, threshold_yen, threshold_li

    if palette is None:
        palette = {0: "seagreen", 1: "salmon"}

    def get_thresh(vals, method):
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 5:
            return np.median(vals)
        if method == "otsu":
            try:
                return threshold_otsu(vals)
            except Exception:
                return np.median(vals)
        elif method == "yen":
            try:
                return threshold_yen(vals)
            except Exception:
                return np.median(vals)
        elif method == "li":
            try:
                return threshold_li(vals)
            except Exception:
                return np.median(vals)
        elif method == "gmm":
            vals_ = vals.reshape(-1, 1)
            try:
                gmm = GaussianMixture(n_components=2, covariance_type='full', n_init=10, random_state=0)
                gmm.fit(vals_)
                means = gmm.means_.flatten()
                if np.abs(np.diff(means)).sum() < 1e-3:
                    # Unimodal, fallback to KDE
                    return get_thresh(vals, "kde")
                grid = np.linspace(vals.min(), vals.max(), 512).reshape(-1, 1)
                post = gmm.predict_proba(grid)
                pos_idx = np.argmax(means)
                diff = post[:, pos_idx] - 0.5
                sign_change = np.where(np.diff(np.sign(diff)))[0]
                if len(sign_change) == 0:
                    return np.median(vals)
                t = grid[sign_change[0]][0]
                return t
            except Exception:
                return np.median(vals)
        elif method == "kde":
            try:
                kde = gaussian_kde(vals)
                grid = np.linspace(vals.min(), vals.max(), 512)
                density = kde(grid)
                peaks, _ = find_peaks(density)
                if len(peaks) < 2:
                    return np.median(vals)
                top2 = np.argsort(density[peaks])[-2:]
                peak_locs = np.sort(peaks[top2])
                valley = np.argmin(density[peak_locs[0]:peak_locs[1]+1]) + peak_locs[0]
                t = grid[valley]
                return t
            except Exception:
                return np.median(vals)
        else:
            raise ValueError(f"Unknown threshold_method: {method}")

    num_files = len(ioi_tracking)
    fig, axes = plt.subplots(1, num_files, figsize=(num_files * figsize_scale, 5), sharex=False, sharey=False, dpi=dpi)
    if num_files == 1:
        axes = np.array([axes])

    for c in range(num_files):
        col_mid_x = (axes[c].get_position().x0 + axes[c].get_position().x1) / 2
        title = f"{ioi_files_name[c]}" if ioi_files_name else f"File {c+1}"
        fig.text(col_mid_x, 1.1, textwrap.fill(title, width=30), fontsize=12, fontweight='bold', ha='center')

    for c, seq in enumerate(ioi_tracking):
        df = ioi_tracking[seq]['ch12'].copy()
        x_key = "raw_intensity_mean_ch1" if use_raw else "intensity_mean_ch1"
        y_key = "raw_intensity_mean_ch2" if use_raw else "intensity_mean_ch2"

        feat = df[[x_key, y_key]].dropna()
        idx = feat.index

        X_plot = np.log1p(feat.values) if use_log else feat.values
        x_vals = X_plot[:, 0]
        y_vals = X_plot[:, 1]

        # Find thresholds for each axis
        x_thresh = get_thresh(x_vals, threshold_method)
        y_thresh = get_thresh(y_vals, threshold_method)

        x_pos = x_vals > x_thresh
        y_pos = y_vals > y_thresh
        group = np.stack([x_pos, y_pos], axis=1).astype(int)
        group_tuples = [tuple(g) for g in group]
        cluster_assign = np.array([group_to_cluster.get(gt, 1) for gt in group_tuples])

        df.loc[idx, "Cell_Status"] = cluster_assign
        if cluster_labels and len(cluster_labels) >= len(set(cluster_assign)):
            df.loc[idx, "Cell_Status_Label"] = [cluster_labels[int(i)] for i in cluster_assign]
        else:
            df.loc[idx, "Cell_Status_Label"] = [f"Cluster {int(i)}" for i in cluster_assign]
        # Assign as Series to avoid ValueError
        df.loc[idx, "Cell_Status_Group"] = pd.Series(list(group_tuples), index=idx)
        ioi_tracking[seq]['ch12'] = df

        ax = axes[c]
        sns.scatterplot(
            x=X_plot[:, 0], y=X_plot[:, 1], hue=cluster_assign,
            palette=palette if len(set(cluster_assign)) == 2 else None, ax=ax, s=point_size, alpha=0.6,
            edgecolor=None, legend=False
        )
        ax.axvline(x_thresh, color="grey", linestyle="--", lw=1)
        ax.axhline(y_thresh, color="grey", linestyle="--", lw=1)

        ax.set_xlim(xlim if xlim else (X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5))
        ax.set_ylim(ylim if ylim else (X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5))

        xlabel = "log-Calcium Green (raw)" if use_log else "Calcium Green (raw)"
        ylabel = "log-TO-PRO-3 (raw)" if use_log else "TO-PRO-3 (raw)"
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

        ax.set_facecolor('#e0f0ff')
        ax.grid(True, color='white', linestyle='-', linewidth=1.2)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Legend
        unique_clusters = sorted(set(cluster_assign))
        handles = [
            plt.Line2D([], [], marker='o', linestyle='', color=palette[i] if palette and i in palette else f"C{i}")
            for i in unique_clusters
        ]
        legend_labels = [cluster_labels[int(i)] if cluster_labels and int(i) < len(cluster_labels) else f"Cluster {i}" for i in unique_clusters]
        ax.legend(handles, legend_labels, title="Cell Type", loc="upper right", fontsize=8)

        print(f"File {c+1} marginal thresholds: x={x_thresh:.2f}, y={y_thresh:.2f} (method: {threshold_method})")

    plt.tight_layout()
    plt.show()

    return ioi_tracking


def threshold_cell_states_3(
    ioi_tracking,
    ioi_files_name=None,
    threshold_method="gmm",
    use_log=True,
    use_raw=True,
    cluster_labels=("Alive", "Dead"),
    palette=None,
    point_size=8,
    xlim=None,
    ylim=None,
    group_to_cluster={(1,0): 0, (0,0): 1, (0,1): 1, (1,1): 1},
    xlabel=None,
    ylabel=None,
    axis_label_fontsize=18,
    legend_title=None,
    legend_fontsize=14,
    hist_bins=40,
    save_path=None,   # e.g. "output_plot"
    save_format=None,  # "svg" or "pdf",
    dpi=300
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    from sklearn.mixture import GaussianMixture
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks
    from skimage.filters import threshold_otsu, threshold_yen, threshold_li

    if palette is None:
        if cluster_labels and len(cluster_labels) == 2:
            palette = {cluster_labels[0]: "seagreen", cluster_labels[1]: "salmon"}
        else:
            palette = None

    def get_thresh(vals, method):
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 5:
            return np.median(vals)
        if method == "otsu":
            try:
                return threshold_otsu(vals)
            except Exception:
                return np.median(vals)
        elif method == "yen":
            try:
                return threshold_yen(vals)
            except Exception:
                return np.median(vals)
        elif method == "li":
            try:
                return threshold_li(vals)
            except Exception:
                return np.median(vals)
        elif method == "gmm":
            vals_ = vals.reshape(-1, 1)
            try:
                gmm = GaussianMixture(n_components=2, covariance_type='full', n_init=10, random_state=0)
                gmm.fit(vals_)
                means = gmm.means_.flatten()
                if np.abs(np.diff(means)).sum() < 1e-3:
                    return get_thresh(vals, "kde")
                grid = np.linspace(vals.min(), vals.max(), 512).reshape(-1, 1)
                post = gmm.predict_proba(grid)
                pos_idx = np.argmax(means)
                diff = post[:, pos_idx] - 0.5
                sign_change = np.where(np.diff(np.sign(diff)))[0]
                if len(sign_change) == 0:
                    return np.median(vals)
                t = grid[sign_change[0]][0]
                return t
            except Exception:
                return np.median(vals)
        elif method == "kde":
            try:
                kde = gaussian_kde(vals)
                grid = np.linspace(vals.min(), vals.max(), 512)
                density = kde(grid)
                peaks, _ = find_peaks(density)
                if len(peaks) < 2:
                    return np.median(vals)
                top2 = np.argsort(density[peaks])[-2:]
                peak_locs = np.sort(peaks[top2])
                valley = np.argmin(density[peak_locs[0]:peak_locs[1]+1]) + peak_locs[0]
                t = grid[valley]
                return t
            except Exception:
                return np.median(vals)
        else:
            raise ValueError(f"Unknown threshold_method: {method}")

    num_files = len(ioi_tracking)
    for c, seq in enumerate(ioi_tracking):
        df = ioi_tracking[seq]['ch12'].copy()
        x_key = "raw_intensity_mean_ch1" if use_raw else "intensity_mean_ch1"
        y_key = "raw_intensity_mean_ch2" if use_raw else "intensity_mean_ch2"

        feat = df[[x_key, y_key]].dropna()
        idx = feat.index

        X_plot = np.log1p(feat.values) if use_log else feat.values
        x_vals = X_plot[:, 0]
        y_vals = X_plot[:, 1]

        # Thresholds
        x_thresh = get_thresh(x_vals, threshold_method)
        y_thresh = get_thresh(y_vals, threshold_method)

        x_pos = x_vals > x_thresh
        y_pos = y_vals > y_thresh
        group = np.stack([x_pos, y_pos], axis=1).astype(int)
        group_tuples = [tuple(g) for g in group]
        cluster_assign = np.array([group_to_cluster.get(gt, 1) for gt in group_tuples])

        if cluster_labels and len(cluster_labels) >= len(set(cluster_assign)):
            status_labels = [cluster_labels[int(i)] for i in cluster_assign]
        else:
            status_labels = [f"Cluster {int(i)}" for i in cluster_assign]

        df.loc[idx, "Cell_Status"] = cluster_assign
        df.loc[idx, "Cell_Status_Label"] = status_labels
        df.loc[idx, "Cell_Status_Group"] = pd.Series(list(group_tuples), index=idx)
        ioi_tracking[seq]['ch12'] = df

        # Prepare for plotting
        plot_df = pd.DataFrame({
            "x_plot": x_vals,
            "y_plot": y_vals,
            "Cell_Status_Label": status_labels
        })

        # --- Custom Joint Plot Layout ---
        fig = plt.figure(figsize=(8, 8), dpi = dpi)
        gs = gridspec.GridSpec(6, 6, hspace=0.05, wspace=0.05)
        ax_scatter = fig.add_subplot(gs[1:, :5])
        ax_histx  = fig.add_subplot(gs[0, :5], sharex=ax_scatter)
        ax_histy  = fig.add_subplot(gs[1:, 5], sharey=ax_scatter)

        # Scatter
        for label in np.unique(status_labels):
            mask = plot_df["Cell_Status_Label"] == label
            ax_scatter.scatter(
                plot_df.loc[mask, "x_plot"],
                plot_df.loc[mask, "y_plot"],
                s=point_size,
                color=palette[label] if palette and label in palette else None,
                label=label,
                alpha=0.7,
                rasterized=True
            )
        x_all = plot_df["x_plot"]
        y_all = plot_df["y_plot"]
        x_edges = np.histogram_bin_edges(x_all, bins=hist_bins)
        y_edges = np.histogram_bin_edges(y_all, bins=hist_bins)

        # Marginals
        for label in np.unique(status_labels):
            mask = plot_df["Cell_Status_Label"] == label
            ax_histx.hist(
                plot_df.loc[mask, "x_plot"], bins=x_edges,
                color=palette[label] if palette and label in palette else None,
                alpha=0.5, histtype='stepfilled', linewidth=1.2
            )
            ax_histy.hist(
                plot_df.loc[mask, "y_plot"], bins=y_edges,
                color=palette[label] if palette and label in palette else None,
                alpha=0.5, histtype='stepfilled', orientation='horizontal', linewidth=1.2
            )

        # Threshold lines
        ax_scatter.axvline(x_thresh, color="grey", linestyle="--", lw=1)
        ax_scatter.axhline(y_thresh, color="grey", linestyle="--", lw=1)
        ax_histx.axvline(x_thresh, color="grey", linestyle="--", lw=1)
        ax_histy.axhline(y_thresh, color="grey", linestyle="--", lw=1)

        # Axis labels
        xlabel_final = xlabel if xlabel else ("log-Calcium Green (raw)" if use_log else "Calcium Green (raw)")
        ylabel_final = ylabel if ylabel else ("log-TO-PRO-3 (raw)" if use_log else "TO-PRO-3 (raw)")
        ax_scatter.set_xlabel(xlabel_final, fontsize=axis_label_fontsize)
        ax_scatter.set_ylabel(ylabel_final, fontsize=axis_label_fontsize)

        # Hide tick labels for marginals
        ax_histx.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax_histy.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax_histx.set_yticklabels([])
        ax_histy.set_xticklabels([])

        # Set limits
        if xlim:
            ax_scatter.set_xlim(xlim)
            ax_histx.set_xlim(xlim)
        if ylim:
            ax_scatter.set_ylim(ylim)
            ax_histy.set_ylim(ylim)

        # Legend
        ax_scatter.legend(
            title=legend_title,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            loc="best"
        )

        # Style
        for spine in ax_scatter.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        for spine in ax_histx.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        for spine in ax_histy.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        ax_scatter.set_facecolor('#e0f0ff')
        ax_scatter.grid(True, color='white', linestyle='-', linewidth=1.2)
        ax_scatter.set_axisbelow(True)

        # Title
        if ioi_files_name:
            fig.suptitle(f"{ioi_files_name[c]}", y=0.98, fontsize=14, fontweight='bold')

        plt.tight_layout()
        if save_path and save_format:
            out_file = f"{save_path}_file{c+1}.{save_format.lower()}"
            fig.savefig(out_file, format=save_format, bbox_inches='tight')
            print(f"Figure saved as: {out_file}")
        plt.show()

        print(f"File {c+1} marginal thresholds: x={x_thresh:.2f}, y={y_thresh:.2f} (method: {threshold_method})")

    return ioi_tracking


def threshold_cell_states_2(
    ioi_tracking,
    ioi_files_name=None,
    threshold_method="gmm",  # "gmm", "kde", "otsu", "yen", "li"
    use_log=True,
    use_raw=True,
    cluster_labels=("Alive", "Dead"),
    dpi=250,
    figsize_scale=5,
    palette=None,
    point_size=15,  # Not used for hist, but kept for API compatibility
    xlim=None,
    ylim=None,
    group_to_cluster={(1,0): 0, (0,0): 1, (0,1): 1, (1,1): 1},
    xlabel=None,
    ylabel=None,
    axis_label_fontsize=18,
    legend_title="Cell_Status_Label",
    legend_fontsize=14,
    hist_square_size=40,  # bins parameter for hist2d
    save_path=None,       # e.g. "output_plot"
    save_format=None      # "svg" or "pdf"
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.mixture import GaussianMixture
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks
    from skimage.filters import threshold_otsu, threshold_yen, threshold_li

    if palette is None:
        if cluster_labels and len(cluster_labels) == 2:
            palette = {cluster_labels[0]: "seagreen", cluster_labels[1]: "salmon"}
        else:
            palette = None

    def get_thresh(vals, method):
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 5:
            return np.median(vals)
        if method == "otsu":
            try:
                return threshold_otsu(vals)
            except Exception:
                return np.median(vals)
        elif method == "yen":
            try:
                return threshold_yen(vals)
            except Exception:
                return np.median(vals)
        elif method == "li":
            try:
                return threshold_li(vals)
            except Exception:
                return np.median(vals)
        elif method == "gmm":
            vals_ = vals.reshape(-1, 1)
            try:
                gmm = GaussianMixture(n_components=2, covariance_type='full', n_init=10, random_state=0)
                gmm.fit(vals_)
                means = gmm.means_.flatten()
                if np.abs(np.diff(means)).sum() < 1e-3:
                    return get_thresh(vals, "kde")
                grid = np.linspace(vals.min(), vals.max(), 512).reshape(-1, 1)
                post = gmm.predict_proba(grid)
                pos_idx = np.argmax(means)
                diff = post[:, pos_idx] - 0.5
                sign_change = np.where(np.diff(np.sign(diff)))[0]
                if len(sign_change) == 0:
                    return np.median(vals)
                t = grid[sign_change[0]][0]
                return t
            except Exception:
                return np.median(vals)
        elif method == "kde":
            try:
                kde = gaussian_kde(vals)
                grid = np.linspace(vals.min(), vals.max(), 512)
                density = kde(grid)
                peaks, _ = find_peaks(density)
                if len(peaks) < 2:
                    return np.median(vals)
                top2 = np.argsort(density[peaks])[-2:]
                peak_locs = np.sort(peaks[top2])
                valley = np.argmin(density[peak_locs[0]:peak_locs[1]+1]) + peak_locs[0]
                t = grid[valley]
                return t
            except Exception:
                return np.median(vals)
        else:
            raise ValueError(f"Unknown threshold_method: {method}")

    num_files = len(ioi_tracking)
    for c, seq in enumerate(ioi_tracking):
        df = ioi_tracking[seq]['ch12'].copy()
        x_key = "raw_intensity_mean_ch1" if use_raw else "intensity_mean_ch1"
        y_key = "raw_intensity_mean_ch2" if use_raw else "intensity_mean_ch2"

        feat = df[[x_key, y_key]].dropna()
        idx = feat.index

        X_plot = np.log1p(feat.values) if use_log else feat.values
        x_vals = X_plot[:, 0]
        y_vals = X_plot[:, 1]

        # Find thresholds for each axis
        x_thresh = get_thresh(x_vals, threshold_method)
        y_thresh = get_thresh(y_vals, threshold_method)

        x_pos = x_vals > x_thresh
        y_pos = y_vals > y_thresh
        group = np.stack([x_pos, y_pos], axis=1).astype(int)
        group_tuples = [tuple(g) for g in group]
        cluster_assign = np.array([group_to_cluster.get(gt, 1) for gt in group_tuples])

        df.loc[idx, "Cell_Status"] = cluster_assign
        if cluster_labels and len(cluster_labels) >= len(set(cluster_assign)):
            df.loc[idx, "Cell_Status_Label"] = [cluster_labels[int(i)] for i in cluster_assign]
        else:
            df.loc[idx, "Cell_Status_Label"] = [f"Cluster {int(i)}" for i in cluster_assign]
        df.loc[idx, "Cell_Status_Group"] = pd.Series(list(group_tuples), index=idx)
        ioi_tracking[seq]['ch12'] = df

        # Prepare DataFrame for plotting
        plot_df = pd.DataFrame({
            "x_plot": x_vals,
            "y_plot": y_vals,
            "Cell_Status_Label": df.loc[idx, "Cell_Status_Label"].values
        })

        # Set axis labels
        xlabel_final = xlabel if xlabel else ("log-Calcium Green (raw)" if use_log else "Calcium Green (raw)")
        ylabel_final = ylabel if ylabel else ("log-TO-PRO-3 (raw)" if use_log else "TO-PRO-3 (raw)")

        # Seaborn jointplot with marginal histograms
        g = sns.jointplot(
            data=plot_df,
            x="x_plot",
            y="y_plot",
            hue="Cell_Status_Label",
            kind="hist",
            marginal_kws=dict(bins=hist_square_size, fill=True, alpha=0.5),
            palette=palette if len(set(cluster_assign)) == 2 else None,
            height=5,
            alpha=0.7
        )
        g.ax_joint.axvline(x_thresh, color="grey", linestyle="--", lw=1)
        g.ax_joint.axhline(y_thresh, color="grey", linestyle="--", lw=1)

        g.set_axis_labels(xlabel_final, ylabel_final, fontsize=axis_label_fontsize)
        g.fig.set_dpi(dpi)
        g.fig.suptitle(f"{ioi_files_name[c]}" if ioi_files_name else f"File {c+1}", y=1.02, fontsize=12, fontweight='bold')

        # Custom legend
        handles, labels = g.ax_joint.get_legend_handles_labels()
        if handles:
            g.ax_joint.legend(
                handles=handles,
                labels=labels,
                title=legend_title,
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
                loc="best"
            )

        for spine in g.ax_joint.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        for spine in g.ax_marg_x.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        for spine in g.ax_marg_y.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)

        g.ax_joint.set_facecolor('#e0f0ff')
        g.ax_joint.grid(True, color='white', linestyle='-', linewidth=1.2)
        g.ax_joint.set_axisbelow(True)

        # Set limits if provided
        if xlim:
            g.ax_joint.set_xlim(xlim)
        if ylim:
            g.ax_joint.set_ylim(ylim)

        print(f"File {c+1} marginal thresholds: x={x_thresh:.2f}, y={y_thresh:.2f} (method: {threshold_method})")

        plt.tight_layout()
        if save_path and save_format:
            out_file = f"{save_path}_file{c+1}.{save_format.lower()}"
            g.fig.savefig(out_file, format=save_format, bbox_inches='tight')
            print(f"Figure saved as: {out_file}")
        plt.show()

    return ioi_tracking

def plot_all_TempoFluo(
    df,
    min_track_length=20,
    ch1_label='721.221 Intensity',
    ch2_label='Death Reporter Intensity',
    ch1_color='seagreen',
    ch2_color='salmon',
    num_cols=10,
    figsize_per_plot=(2, 2.5)
):
    """
    Plot raw fluorescence traces (channel 1 & 2) for each particle in a grid layout.

    Parameters:
        df (pd.DataFrame): Dataframe containing tracking data with at least ['id', 't', 'raw_intensity_mean_ch1', 'raw_intensity_mean_ch2'] columns.
        min_track_length (int): Minimum number of frames required to include a particle.
        ch1_label (str): Legend label for channel 1 (e.g. live reporter).
        ch2_label (str): Legend label for channel 2 (e.g. death reporter).
        ch1_color (str): Line colour for channel 1.
        ch2_color (str): Line colour for channel 2.
        num_cols (int): Number of columns in the subplot grid.
        figsize_per_plot (tuple): (width, height) per subplot in inches.
    """
    df = df.groupby('id').filter(lambda sub_df: len(sub_df) >= min_track_length)
    unique_particles = df['id'].unique()
    num_particles = len(unique_particles)

    num_rows = int(np.ceil(num_particles / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(num_cols * figsize_per_plot[0], num_rows * figsize_per_plot[1]),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, particle in tqdm(enumerate(unique_particles), total=num_particles):
        sub_df = df[df['id'] == particle]
        ax = axes[idx]
        ax.plot(sub_df['t'], sub_df['raw_intensity_mean_ch1'], marker='o', linestyle='-', color=ch1_color, label=ch1_label)
        ax.plot(sub_df['t'], sub_df['raw_intensity_mean_ch2'], marker='s', linestyle='--', color=ch2_color, label=ch2_label)
        ax.set_title(f"ID {particle}", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        if idx < (num_rows - 1) * num_cols:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Frame")
        if idx % num_cols != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Fluorescence Intensity")

    for j in range(idx + 1, num_rows * num_cols):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()
    

def categorise(df, min_track_length=25):
    df = df.groupby('id').filter(lambda sub_df: len(sub_df) >= min_track_length)
    unique_particles = df['id'].unique()

    always_0 = []
    always_1 = []
    transition_0_to_1 = []

    for particle_id in unique_particles:
        sub_df = df[df['id'] == particle_id].sort_values(by='t')
        status_array = np.array(sub_df['Cell_Status'])

        if np.all(status_array == 0):
            always_0.append(particle_id)
        elif np.all(status_array == 1):
            always_1.append(particle_id)
        elif np.any(status_array == 1):
            first_one_idx = np.where(status_array == 1)[0][0]
            if np.all(status_array[:first_one_idx] == 0) and np.all(status_array[first_one_idx:] == 1):
                transition_0_to_1.append(particle_id)

    return always_0, always_1, transition_0_to_1


def plot_ex_TempoFluo(
    df,
    always_0, always_1, transition_0_to_1,
    num_cols=5,
    figsize=(15, 8),
    ch1_label='721.221',
    ch2_label='TO-PRO-3',
    ch1_color='seagreen',
    ch2_color='salmon',
    row_titles=None,
    row_title_fontsize=10,
    show_ids=True,
    custom_titles=None,
    xlabel="Frames",
    ylabel="Fluorescence Intensity",
    xlabel_fontsize=18,
    ylabel_fontsize=18,
    legend_fontsize=14,
    profile_linewidth=2,
    mark_size=3,
    ylim=None,                      # (0, 2000)
    save_path=None                  # e.g. "temporal_profiles.svg"
):
    categories = [always_0, always_1, transition_0_to_1]
    default_titles = [None, None, None]
    titles = row_titles if row_titles is not None else default_titles
    num_rows = 3

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True)
    axes = np.array(axes).reshape(num_rows, num_cols)

    title_counter = 0
    for row_idx, cat in enumerate(categories):
        for col_idx in range(num_cols):
            ax = axes[row_idx, col_idx]
            if ylim is not None:
                ax.set_ylim(ylim)

            if col_idx < len(cat):
                pid = cat[col_idx]
                sub_df = df[df['id'] == pid]
                ax.plot(
                    sub_df['t'], sub_df['raw_intensity_mean_ch1'],
                    'o-', color=ch1_color, label=ch1_label,
                    linewidth=profile_linewidth, markersize=mark_size
                )
                ax.plot(
                    sub_df['t'], sub_df['raw_intensity_mean_ch2'],
                    's-', color=ch2_color, label=ch2_label,
                    linewidth=profile_linewidth, markersize=mark_size
                )
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.xaxis.set_major_locator(MultipleLocator(10))

                if show_ids:
                    if custom_titles and title_counter < len(custom_titles):
                        ax.set_title(custom_titles[title_counter], fontsize=8)
                    else:
                        ax.set_title(f"ID {pid}", fontsize=8)
                title_counter += 1
            else:
                ax.axis('off')

            if col_idx == 0:
                ax.set_ylabel(titles[row_idx], fontsize=row_title_fontsize)

            if row_idx == num_rows - 1:
                ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)

    fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical', fontsize=ylabel_fontsize)

    handles = [
        plt.Line2D([], [], marker='o', linestyle='-', color=ch1_color, label=ch1_label),
        plt.Line2D([], [], marker='s', linestyle='-', color=ch2_color, label=ch2_label)
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=legend_fontsize)
    fig.tight_layout(rect=[0.05, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, format=save_path.split('.')[-1], bbox_inches='tight')
        print(f"Figure saved as: {save_path}")

    plt.show()

def plot_barrels(
    df,
    always_0, always_1, transition_0_to_1,
    transition_col='Cell_Status',
    time_col='t',
    ch1='raw_intensity_mean_ch1',
    ch2='raw_intensity_mean_ch2',
    ch1_label='Calcein Green AM (mean)',
    ch2_label='TO-PRO-3 (mean)',
    ch1_color='seagreen',
    ch2_color='salmon',
    figsize=(18, 5),
    alpha=0.3,
    linewidth=1.8,
    grid_alpha=0.4,
    xlims=None,        # [(0, 60), (0, 60), (-50, 50)]
    titles=None,       # ["Live", "Dead", "Transition"]
    save_path=None     # "barrels_plot.pdf"
):
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
    axes = np.array(axes)

    if titles is None:
        titles = ["Always Live", "Always Dead", "Live → Dead Transition"]
    if xlims is None:
        xlims = [None, None, None]

    def plot_mean_std(ax, t, ch1_mean, ch1_std, ch2_mean, ch2_std, title, xlim=None, rel=False):
        ax.plot(t, ch1_mean, label=ch1_label, color=ch1_color, linewidth=linewidth)
        ax.fill_between(t, ch1_mean - ch1_std, ch1_mean + ch1_std, color=ch1_color, alpha=alpha)

        ax.plot(t, ch2_mean, label=ch2_label, color=ch2_color, linewidth=linewidth)
        ax.fill_between(t, ch2_mean - ch2_std, ch2_mean + ch2_std, color=ch2_color, alpha=alpha)

        if rel:
            ax.axvline(0, color='k', linestyle=':', label='Death Point')

        if xlim:
            ax.set_xlim(xlim)

        ax.set_xlabel("Frame" if not rel else "Relative Frame (transition = 0)")
        ax.set_title(title, fontsize=12)
        ax.grid(True, linestyle='--', alpha=grid_alpha)

    # Panel 1: Always Live
    df_live = df[df['id'].isin(always_0)]
    summary_live = df_live.groupby(time_col).agg({ch1: ['mean', 'std'], ch2: ['mean', 'std']}).reset_index()
    plot_mean_std(
        axes[0],
        summary_live[time_col],
        summary_live[(ch1, 'mean')],
        summary_live[(ch1, 'std')],
        summary_live[(ch2, 'mean')],
        summary_live[(ch2, 'std')],
        title=titles[0],
        xlim=xlims[0]
    )

    # Panel 2: Always Dead
    df_dead = df[df['id'].isin(always_1)]
    summary_dead = df_dead.groupby(time_col).agg({ch1: ['mean', 'std'], ch2: ['mean', 'std']}).reset_index()
    plot_mean_std(
        axes[1],
        summary_dead[time_col],
        summary_dead[(ch1, 'mean')],
        summary_dead[(ch1, 'std')],
        summary_dead[(ch2, 'mean')],
        summary_dead[(ch2, 'std')],
        title=titles[1],
        xlim=xlims[1]
    )

    # Panel 3: Transition
    rel_profiles = []
    for pid in transition_0_to_1:
        sub_df = df[df['id'] == pid].sort_values(by=time_col)
        transition_idx = np.argmax(sub_df[transition_col].values == 1)
        rel_frame = np.arange(len(sub_df)) - transition_idx
        temp_df = pd.DataFrame({
            'rel_frame': rel_frame,
            ch1: sub_df[ch1].values,
            ch2: sub_df[ch2].values
        })
        rel_profiles.append(temp_df)

    rel_df = pd.concat(rel_profiles)
    summary_rel = rel_df.groupby('rel_frame').agg({ch1: ['mean', 'std'], ch2: ['mean', 'std']}).reset_index()

    plot_mean_std(
        axes[2],
        summary_rel['rel_frame'],
        summary_rel[(ch1, 'mean')],
        summary_rel[(ch1, 'std')],
        summary_rel[(ch2, 'mean')],
        summary_rel[(ch2, 'std')],
        title=titles[2],
        xlim=xlims[2],
        rel=True
    )

    axes[0].set_ylabel("Fluorescence Intensity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        fig.savefig(save_path, format=save_path.split('.')[-1], bbox_inches='tight')
        print(f"Figure saved as: {save_path}")

    plt.show()




def animate_scProfile(
    ioi_tracking,
    ioi_files,
    file=0,
    target_Tu_id=16,
    time_interval=None,
    crop_size=60,
    fig_width=16,
    fig_height=6,
    profile_time_extend=0,
    save_gif_path=None,
    ch1_label='Calcein Green AM',
    ch2_label='TO-PRO-3'
):
    """
    Animate a single cell trajectory with its fluorescence intensity profile.

    Parameters
    ----------
    ioi_tracking : list of dict
        Output from tracking, each dict contains DataFrame(s) per channel set.
    ioi_files : list of np.ndarray
        Time-lapse image stacks with shape [T, H, W, C].
    file : int
        Index to select sample.
    target_Tu_id : int
        ID of the cell to animate.
    time_interval : int or None
        Number of frames to animate (default is total frame count).
    crop_size : int
        Size of the cropped image centred on the cell.
    fig_width : int
        Width of the figure.
    fig_height : int
        Height of the figure.
    profile_time_extend : int
        Number of frames to extend profile window (left/right).
    save_gif_path : str or None
        If set, path to save the gif. Otherwise shows inline.
    ch1_label : str
        Label for channel 1 (e.g., Calcein).
    ch2_label : str
        Label for channel 2 (e.g., TO-PRO-3).
    """

    def merge_view(image1, image2, image3, brightfield, 
                   blue_brightness_factor=0.75, 
                   green_brightness_factor=1.2, 
                   red_brightness_factor=0.6, 
                   brightfield_intensity_factor=1):
        def normalize_and_scale(image, brightness_factor):
            if np.max(image) > 0:
                normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
            else:
                normalized = image
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

    def pad_crop(img_crop, crop_size, cy, cx):
        y, x, _ = img_crop.shape
        pad_y1 = max(0, (crop_size//2) - cy)
        pad_y2 = max(0, crop_size - y - pad_y1)
        pad_x1 = max(0, (crop_size//2) - cx)
        pad_x2 = max(0, crop_size - x - pad_x1)
        return np.pad(img_crop, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0,0)), 'constant')

    df = ioi_tracking[file]['ch12']
    df = df.groupby('id').filter(lambda sub_df: len(sub_df) >= 25)
    target_df = df[df['id'] == target_Tu_id].sort_values('t')
    im = ioi_files[file]

    if time_interval is None:
        time_interval = im.shape[0]

    half_crop = crop_size // 2

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=500)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])
    ax0 = fig.add_subplot(gs[0])
    ax0.set_title("Raw Image (Target Cell Centered)", fontsize=14, fontweight="bold")
    ax0.set_aspect('equal')
    ax1 = fig.add_subplot(gs[1])
    ax1.set_title(f"Intensity Profile (Particle {target_Tu_id})", fontsize=12)
    line_ch1, = ax1.plot(target_df['t'], target_df['raw_intensity_mean_ch1'], 'o-', color='seagreen', label=ch1_label)
    line_ch2, = ax1.plot(target_df['t'], target_df['raw_intensity_mean_ch2'], 's--', color='salmon', label=ch2_label)
    current_marker_ch1, = ax1.plot([], [], 'o', color='darkgreen', markersize=10)
    current_marker_ch2, = ax1.plot([], [], 's', color='firebrick', markersize=10)
    ax1.legend(loc='upper right')
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Fluorescence Intensity")
    ax1.grid(True, linestyle="--", alpha=0.5)

    t_min = max(target_df['t'].min() - profile_time_extend, 0)
    t_max = min(target_df['t'].max() + profile_time_extend, im.shape[0]-1)
    ax1.set_xlim(t_min, t_max)
    ymin = min(target_df['raw_intensity_mean_ch1'].min(), target_df['raw_intensity_mean_ch2'].min())
    ymax = max(target_df['raw_intensity_mean_ch1'].max(), target_df['raw_intensity_mean_ch2'].max())
    ax1.set_ylim(ymin - 0.1*(ymax-ymin), ymax + 0.1*(ymax-ymin))
    vertical_line = Line2D([0, 0], [ymin, ymax], color="dodgerblue", linestyle="--", linewidth=2, animated=True)
    ax1.add_line(vertical_line)

    traj_line, = ax0.plot([], [], 'o-', color='navy', markersize=4, alpha=0.7, animated=True)
    center_marker, = ax0.plot([], [], 'o', markersize=10, alpha=0.6, animated=True)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_xlim(0, crop_size)
    ax0.set_ylim(0, crop_size)

    def get_crop_and_traj(frame_idx):
        if frame_idx in target_df['t'].values:
            curr = target_df[target_df['t'] == frame_idx]
            center_x = int(round(curr['x'].values[0]))
            center_y = int(round(curr['y'].values[0]))
        else:
            before = target_df[target_df['t'] < frame_idx]
            if len(before):
                center_x = int(round(before['x'].values[-1]))
                center_y = int(round(before['y'].values[-1]))
            else:
                center_x = int(round(target_df['x'].values[0]))
                center_y = int(round(target_df['y'].values[0]))
        y1 = max(0, center_y - half_crop)
        y2 = min(im.shape[1], center_y + half_crop)
        x1 = max(0, center_x - half_crop)
        x2 = min(im.shape[2], center_x + half_crop)
        cy = center_y - y1
        cx = center_x - x1
        img_crop = merge_view(im[frame_idx, y1:y2, x1:x2, 0],
                              im[frame_idx, y1:y2, x1:x2, 1],
                              im[frame_idx, y1:y2, x1:x2, 2],
                              im[frame_idx, y1:y2, x1:x2, 3])
        img_crop = pad_crop(img_crop, crop_size, cy, cx)
        traj = target_df[target_df['t'] <= frame_idx]
        x_vals = traj['x'].values - center_x + crop_size // 2
        y_vals = traj['y'].values - center_y + crop_size // 2
        return img_crop, x_vals, y_vals, crop_size // 2, crop_size // 2

    img_crop0, x_vals0, y_vals0, cx0, cy0 = get_crop_and_traj(0)
    im_raw = ax0.imshow(img_crop0, animated=True, origin='upper', aspect='equal')
    traj_line.set_data(x_vals0, y_vals0)
    center_marker.set_data([cx0], [cy0])

    def update(frame_idx):
        updated = []
        img_crop, x_vals, y_vals, cx, cy = get_crop_and_traj(frame_idx)
        im_raw.set_array(img_crop)
        traj_line.set_data(x_vals, y_vals)
        center_marker.set_data([cx], [cy])
        updated.extend([im_raw, traj_line, center_marker])
        if frame_idx in target_df['t'].values:
            idx = target_df.index[target_df['t'] == frame_idx][0]
            t_val = target_df.loc[idx, 't']
            ch1_val = target_df.loc[idx, 'raw_intensity_mean_ch1']
            ch2_val = target_df.loc[idx, 'raw_intensity_mean_ch2']
            current_marker_ch1.set_data([t_val], [ch1_val])
            current_marker_ch2.set_data([t_val], [ch2_val])
            vertical_line.set_data([t_val, t_val], [ymin, ymax])
        else:
            current_marker_ch1.set_data([], [])
            current_marker_ch2.set_data([], [])
            vertical_line.set_data([frame_idx, frame_idx], [ymin, ymax])
        updated.extend([current_marker_ch1, current_marker_ch2, vertical_line])
        return updated

    anim = FuncAnimation(fig, update, frames=time_interval, interval=5, blit=True)
    plt.close(fig)

    if save_gif_path is not None:
        import os
        gif_path = os.path.join(save_gif_path, f"animation_Tu_{target_Tu_id}.gif")
        anim.save(gif_path, writer=PillowWriter(fps=2), dpi=300)
        print(f"GIF saved to {gif_path}")
        return HTML(anim.to_jshtml())
    else:
        return HTML(anim.to_jshtml())