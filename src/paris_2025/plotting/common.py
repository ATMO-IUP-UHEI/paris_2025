import datetime
import sys

import matplotlib.pyplot as plt


def get_metadata(description=None):
    """Get metadata for the plots."""
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    caller_name = sys._getframe().f_back.f_code.co_name  # type: ignore
    _description = f"Created by function '{caller_name}' on {date_str}."
    if description is not None:
        _description += f"\n{description}"
    return {"Description": _description}


def save_table_as_png(df, filename, caption="", figsize=None):
    """Save a DataFrame as a PNG with background gradient styling similar to pandas
    style.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save
    filename : str
        Output filename (should end with .png)
    caption : str
        Table caption
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated.
    """
    # Normalize values column-wise for color mapping
    # Choose colormap separately for each column
    normalized = df.copy()
    colormaps = {}

    for col in df.columns:
        col_data = df[col].values
        has_negative = (col_data < 0).any()

        # Choose colormap based on data in this column
        if has_negative:
            colormaps[col] = plt.get_cmap("RdBu_r")  # Diverging colormap
            # For diverging colormap, normalize symmetrically around 0
            max_abs = max(abs(col_data.min()), abs(col_data.max()))
            if max_abs > 0:
                normalized[col] = (col_data + max_abs) / (2 * max_abs)
            else:
                normalized[col] = 0.5
        else:
            colormaps[col] = plt.get_cmap("Blues")  # Sequential colormap
            # For sequential colormap, normalize from min to max
            if col_data.max() != col_data.min():
                normalized[col] = (col_data - col_data.min()) / (
                    col_data.max() - col_data.min()
                )
            else:
                normalized[col] = 0.5

    # Auto-calculate figure size if not provided
    if figsize is None:
        n_rows, n_cols = df.shape
        figsize = (max(8, n_cols * 1.5), max(4, n_rows * 0.4 + 1))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("tight")
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style cells with background gradient
    for i in range(len(df)):
        for j, col in enumerate(df.columns):
            cell = table[(i + 1, j)]
            cm = colormaps[col]
            color = cm(normalized.iloc[i, j])
            cell.set_facecolor(color)
            cell.set_text_props(weight="normal")

    # Style header
    for j in range(len(df.columns)):
        cell = table[(0, j)]
        cell.set_facecolor("#40466e")
        cell.set_text_props(weight="bold", color="white")

    # Style row labels
    for i in range(len(df)):
        cell = table[(i + 1, -1)]
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(weight="bold")

    # Add caption
    if caption:
        plt.title(caption, fontsize=12, weight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Table saved to {filename}")
