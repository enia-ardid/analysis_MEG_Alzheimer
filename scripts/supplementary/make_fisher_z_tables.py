#!/usr/bin/env python3
from __future__ import annotations

"""Generate clean LaTeX and table-figure outputs for Fisher-z supplementary results."""

import argparse
import os
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import pandas as pd


CAPTION_COMPONENT = (
    "Comparación a nivel de componentes entre el promedio directo en escala r y el promedio en espacio Fisher-z."
)
CAPTION_ENDPOINT = (
    "Comparación a nivel de endpoints entre el promedio directo en escala r y el promedio en espacio Fisher-z."
)

COMPONENT_FIGURE_HEADERS = [
    "Component",
    "Family",
    "Band",
    "Definition",
    "Mean\nsubject r",
    "Min\nsubject r",
    "Max\nsubject r",
    "Mean |Δ|",
    "Median |Δ|",
    "Max |Δ|",
]

ENDPOINT_FIGURE_HEADERS = [
    "Endpoint",
    "Pearson r",
    "Mean |Δ|",
    "Max |Δ|",
    "Δ(C-NC)\nmean-r",
    "Δ(C-NC)\nFisher-z",
    "Cohen d\nmean-r",
    "Cohen d\nFisher-z",
    "p1\nmean-r",
    "p1\nFisher-z",
    "Same\nsign",
    "Same nominal\nsignificance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate supplementary Fisher-z LaTeX tables and optional table-figure exports."
    )
    parser.add_argument(
        "--input-dir",
        default="supplementary/source_tables",
        help="Directory containing table_fisher_z_component_bias.csv and table_fisher_z_endpoint_comparison.csv.",
    )
    parser.add_argument(
        "--supplementary-dir",
        default="supplementary/tables",
        help="Output directory for LaTeX tables.",
    )
    parser.add_argument(
        "--figure-dir",
        default="supplementary/figures",
        help="Output directory for PDF/PNG table figures.",
    )
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 12,
            "axes.titlesize": 15,
            "figure.titlesize": 16,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def require_csv(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"Missing required CSV input: {path}")
    return path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def escape_latex(text: object) -> str:
    value = str(text)
    replacements = {
        "\\": r"\\textbackslash{}",
        "_": r"\\_",
        "%": r"\\%",
        "&": r"\\&",
        "#": r"\\#",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def sci_to_latex(value: float) -> str:
    if value == 0:
        return "0"
    formatted = f"{value:.2e}"
    mantissa, exponent = formatted.split("e")
    return rf"${mantissa} \\times 10^{{{int(exponent)}}}$"


def fmt_value_latex(column: str, value: object) -> str:
    lower = column.lower()
    if "same sign" in lower or "same nominal significance" in lower:
        return escape_latex(value)
    if isinstance(value, str):
        return escape_latex(value)
    if pd.isna(value):
        return ""
    numeric = float(value)
    if "pearson" in lower:
        return f"{numeric:.6f}"
    if lower.startswith("p"):
        return f"{numeric:.4f}"
    if "mean |δ|" in lower or "median |δ|" in lower or "max |δ|" in lower:
        return sci_to_latex(numeric)
    if "Δ(" in column or "Cohen d" in column or "Mean subject r" in column or "Min subject r" in column or "Max subject r" in column:
        return f"{numeric:.4f}"
    return escape_latex(value)


def fmt_value_display(column: str, value: object) -> str:
    lower = column.lower()
    if "same sign" in lower or "same nominal significance" in lower:
        return str(value)
    if isinstance(value, str):
        return str(value)
    if pd.isna(value):
        return ""
    numeric = float(value)
    if "pearson" in lower:
        return f"{numeric:.6f}"
    if lower.startswith("p"):
        return f"{numeric:.4f}"
    if "mean |δ|" in lower or "median |δ|" in lower or "max |δ|" in lower:
        return f"{numeric:.2e}"
    if "Δ(" in column or "Cohen d" in column or "Mean subject r" in column or "Min subject r" in column or "Max subject r" in column:
        return f"{numeric:.4f}"
    return str(value)


def build_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    headers = [escape_latex(col) for col in df.columns]
    body_lines = []
    for row in df.itertuples(index=False, name=None):
        cells = [fmt_value_latex(column, value) for column, value in zip(df.columns, row)]
        body_lines.append("        " + " & ".join(cells) + r" \\")

    column_spec = "l" + "c" * (len(df.columns) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        rf"    \caption{{{caption}}}",
        rf"    \label{{{label}}}",
        r"    \resizebox{\textwidth}{!}{%",
        rf"    \begin{{tabular}}{{{column_spec}}}",
        r"        \toprule",
        "        " + " & ".join(headers) + r" \\",
        r"        \midrule",
        *body_lines,
        r"        \bottomrule",
        r"    \end{tabular}%",
        r"    }",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def write_latex_table(df: pd.DataFrame, caption: str, label: str, output_path: Path) -> None:
    output_path.write_text(build_latex_table(df, caption=caption, label=label))


def _wrap_value(value: str, width: int) -> str:
    return textwrap.fill(value, width=width, break_long_words=False, break_on_hyphens=False)


def build_table_figure(
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    output_stem: Path,
    header_labels: list[str],
    wrap_widths: dict[str, int],
    figure_width: float,
    font_size: float,
) -> None:
    display = df.copy()
    if "Definition" in display.columns:
        display["Definition"] = display["Definition"].replace(
            {
                "TempPar full composite component": "TempPar full component",
                "TempPar inter-network component": "TempPar inter-network component",
            }
        )
    for column in display.columns:
        values = [fmt_value_display(column, value) for value in display[column]]
        if column in wrap_widths:
            values = [_wrap_value(value, wrap_widths[column]) for value in values]
        display[column] = values

    n_rows = len(display) + 1
    fig_height = max(2.8, 0.70 * n_rows + 1.0)
    fig, ax = plt.subplots(figsize=(figure_width, fig_height))
    ax.axis("off")
    bbox_top = 0.87
    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985)
    if subtitle:
        fig.text(
            0.01,
            0.945,
            subtitle,
            ha="left",
            va="bottom",
            fontsize=10.3,
            color="#444444",
        )
    if not title and not subtitle:
        bbox_top = 1.0

    table = ax.table(
        cellText=display.values,
        colLabels=header_labels,
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, bbox_top],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.40)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#b9c2cc")
        if row == 0:
            cell.set_facecolor("#eef2f6")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8fafc")

    fig.savefig(output_stem.with_suffix(".png"))
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_style()

    input_dir = (REPO_ROOT / args.input_dir).resolve()
    supplementary_dir = (REPO_ROOT / args.supplementary_dir).resolve()
    figure_dir = (REPO_ROOT / args.figure_dir).resolve()

    component_csv = require_csv(input_dir / "table_fisher_z_component_bias.csv")
    endpoint_csv = require_csv(input_dir / "table_fisher_z_endpoint_comparison.csv")

    ensure_dir(supplementary_dir)
    ensure_dir(figure_dir)

    component_df = pd.read_csv(component_csv)
    endpoint_df = pd.read_csv(endpoint_csv)

    component_tex = supplementary_dir / "table_fisher_z_component_bias.tex"
    endpoint_tex = supplementary_dir / "table_fisher_z_endpoint_comparison.tex"

    write_latex_table(
        component_df,
        caption=CAPTION_COMPONENT,
        label="tab:fisher_z_component_bias_supplementary",
        output_path=component_tex,
    )
    write_latex_table(
        endpoint_df,
        caption=CAPTION_ENDPOINT,
        label="tab:fisher_z_endpoint_comparison_supplementary",
        output_path=endpoint_tex,
    )

    build_table_figure(
        component_df,
        title="",
        subtitle="",
        output_stem=figure_dir / "fig_fisher_z_component_table",
        header_labels=COMPONENT_FIGURE_HEADERS,
        wrap_widths={"Component": 18, "Definition": 18},
        figure_width=17.0,
        font_size=9.8,
    )
    build_table_figure(
        endpoint_df,
        title="",
        subtitle="",
        output_stem=figure_dir / "fig_fisher_z_endpoint_table",
        header_labels=ENDPOINT_FIGURE_HEADERS,
        wrap_widths={"Endpoint": 14},
        figure_width=18.5,
        font_size=9.3,
    )

    print("Generated:")
    print(component_tex)
    print(endpoint_tex)
    print(figure_dir / "fig_fisher_z_component_table.pdf")
    print(figure_dir / "fig_fisher_z_component_table.png")
    print(figure_dir / "fig_fisher_z_endpoint_table.pdf")
    print(figure_dir / "fig_fisher_z_endpoint_table.png")
    print("\nCommands to regenerate:")
    print("python scripts/supplementary/make_fisher_z_tables.py")
    print("python scripts/supplementary/make_fisher_z_tables.py --input-dir supplementary/source_tables --supplementary-dir supplementary/tables --figure-dir supplementary/figures")


if __name__ == "__main__":
    main()
