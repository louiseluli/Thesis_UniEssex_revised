# -*- coding: utf-8 -*-
"""
academic_tables.py

Purpose
-------
Convert pandas DataFrames into publication-quality LaTeX tables with consistent
styling and safe defaults. This utility centralizes table generation so that all
chapters share consistent formatting and captions/labels are reproducible.

Why this matters
----------------
Professional presentation is part of the assessment. Centralizing LaTeX export
minimizes format drift and makes the dissertation easy to rebuild.

Creates
-------
- One .tex file (complete \\begin{table} ... \\end{table}) per call in
  `dissertation/auto_tables/`.

"""

import os
import pandas as pd
import time

# --- 1. Core Table Generation Function ---

def dataframe_to_latex_table(
    df: pd.DataFrame,
    save_path: str,
    caption: str,
    label: str,
    note: str = None,
    precision: int = 3
):
    """
    Convert a pandas DataFrame into a complete LaTeX table file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the results to be tabulated.
    save_path : str
        Full path for the .tex file, e.g. '.../dissertation/auto_tables/my_table.tex'.
    caption : str
        Caption to display above the table.
    label : str
        LaTeX label for cross-referencing, e.g. 'tab:results'.
    note : Optional[str], default None
        Optional note (definitions, significance levels) placed under the table.
    precision : int, default 3
        Decimal places for floating-point numbers.

    Returns
    -------
    None
        Writes a .tex file to `save_path`. Prints timing info.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame.

    Notes
    -----
    - Index column and data columns are left-aligned to avoid overflow on narrow pages.
    - Special LaTeX characters are escaped; missing values are rendered as '-'.
    - The table is wrapped in a complete \\begin{table} environment ready for inclusion.
    """
    t0 = time.perf_counter()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    print(f"Generating LaTeX table for: {label}...")

    # --- Create LaTeX String ---
    # Use pandas' built-in .to_latex() method as a starting point.
    # 'cl' aligns the first column (index) to the center, and the rest to the left.
    num_cols = len(df.columns)
    col_format = 'l' * num_cols # Left-align all data columns
    
    latex_string = df.to_latex(
        index=True,          # Assume the index is meaningful (e.g., group names)
        header=True,
        float_format=f"%.{precision}f",
        column_format=f"l{col_format}", # Center-align index, left-align data
        escape=True,         # Escape special LaTeX characters like % and &
        na_rep='-'           # Representation for missing values
    )

    # --- Wrap in a complete LaTeX table environment ---
    latex_full = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        latex_string,
    ]

    if note:
        latex_full.append(f"\\begin{{tablenotes}}[flushleft]")
        latex_full.append(f"\\item \\small{{{note}}}")
        latex_full.append(f"\\end{{tablenotes}}")

    latex_full.append("\\end{table}")

    final_latex_code = "\n".join(latex_full)

    # --- Save to file ---
    try:
        # Ensure the directory exists
        output_dir = os.path.dirname(save_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(save_path, 'w') as f:
            f.write(final_latex_code)
        
        # Print the full path of the created artifact
        print(f"✓ Artefact saved: {os.path.abspath(save_path)}")

    except Exception as e:
        print(f"✗ ERROR: Could not write LaTeX table to {save_path}. Reason: {e}")
    finally:
      dt = time.perf_counter() - t0
      print(f"[TIME] academic_tables.dataframe_to_latex_table runtime: {dt:.2f}s")


# --- 2. Example Usage (for demonstration) ---
if __name__ == '__main__':
    print("Running academic_tables.py example...")

    # 1. Create a sample DataFrame
    sample_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Random Forest': [0.8512, 0.8345, 0.8819, 0.8576],
        'BERT': [0.9256, 0.9133, 0.9401, 0.9265]
    }
    df_sample = pd.DataFrame(sample_data).set_index('Metric')

    # 2. Define parameters for the table
    example_save_path = os.path.join(
        'dissertation', 'auto_tables', 'model_performance_comparison.tex'
    )
    example_caption = "Comparison of Baseline Model Performance Metrics."
    example_label = "tab:model-performance"
    example_note = "All metrics evaluated on the held-out test set. Best values are not highlighted."

    # 3. Call the function
    dataframe_to_latex_table(
        df=df_sample,
        save_path=example_save_path,
        caption=example_caption,
        label=example_label,
        note=example_note
    )

    print("\nExample LaTeX table generation complete.")
    print(f"Please check the generated file at: {example_save_path}")