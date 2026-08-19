"""Design tokens and shared styling for the terminal-grade dark interface.

Single source of truth for colour, typography and chart styling so the
Streamlit layer and the matplotlib layer can never drift apart.

Design constraints enforced here:
  - flat surfaces only: no gradients, no glassmorphism, no glow shadows
  - 4px maximum corner radius
  - dense typography: 11-12px labels, tabular numerals on all figures
  - rigid 16px grid gutters
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
BG = "#121212"           # application background
SURFACE = "#1A1A1A"      # cards, panels, sidebar, chart canvas
SURFACE_ALT = "#1E1E1E"  # nested surfaces, table headers, hover states
BORDER = "#2A2A2A"       # dividers and card borders
BORDER_STRONG = "#333333"  # focus and active borders

TEXT = "#F3F4F6"         # values and headers
TEXT_MUTED = "#9CA3AF"   # labels, dates, secondary copy
TEXT_DIM = "#6B7280"     # axis ticks, tertiary copy

GREEN = "#10B981"        # profit / pass
RED = "#EF4444"          # loss / breach
BLUE = "#3B82F6"         # brand and primary chart accent
AMBER = "#F59E0B"        # caution / review

GRID = "#242424"         # hairline chart grid

# Flat, opaque fill tints for chart areas - solid colours, never alpha glows.
BLUE_FILL = "#22334F"
RED_FILL = "#3A2020"
GREEN_FILL = "#172E25"
AMBER_FILL = "#332815"

RADIUS = "4px"
GUTTER = "16px"

FONT_STACK = (
    "Inter, Roboto, system-ui, -apple-system, 'Segoe UI', "
    "'Helvetica Neue', Arial, sans-serif"
)

# Semantic tone -> (accent colour, flat chart fill)
TONES = {
    "positive": (GREEN, GREEN_FILL),
    "negative": (RED, RED_FILL),
    "warning": (AMBER, AMBER_FILL),
    "neutral": (BLUE, BLUE_FILL),
    "muted": (TEXT_MUTED, SURFACE_ALT),
}


def tone_color(tone):
    """Accent colour for a semantic tone name."""
    return TONES.get(tone, TONES["neutral"])[0]


def tone_fill(tone):
    """Flat chart fill for a semantic tone name."""
    return TONES.get(tone, TONES["neutral"])[1]


def pnl_tone(value):
    """Tone for a signed P&L-style number."""
    return "positive" if value >= 0 else "negative"


# --------------------------------------------------------------------------
# Streamlit styling
# --------------------------------------------------------------------------
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: %(bg)s;
    --surface: %(surface)s;
    --surface-alt: %(surface_alt)s;
    --border: %(border)s;
    --border-strong: %(border_strong)s;
    --text: %(text)s;
    --muted: %(muted)s;
    --dim: %(dim)s;
    --green: %(green)s;
    --red: %(red)s;
    --blue: %(blue)s;
    --amber: %(amber)s;
    --radius: %(radius)s;
    --gutter: %(gutter)s;
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    font-family: %(font)s;
    font-feature-settings: "tnum" 1;
}

.stApp, [data-testid="stAppViewContainer"] { background-color: var(--bg); }
[data-testid="stHeader"] { background-color: var(--bg); }
[data-testid="stToolbar"] { background-color: var(--bg); }

.block-container {
    /* Streamlit's toolbar is position:absolute and 3.75rem tall, so reserve
       room for it or the app header renders underneath the Stop/Fork buttons. */
    padding-top: calc(3.75rem + 12px);
    padding-bottom: 2.5rem;
    max-width: 1600px;
}

/* Rigid grid: uniform 16px gutters everywhere */
[data-testid="stHorizontalBlock"] { gap: var(--gutter); }
[data-testid="stVerticalBlock"] { gap: var(--gutter); }
[data-testid="column"] { gap: var(--gutter); }

/* No glow shadows anywhere in the chrome */
[data-testid="stMetric"],
[data-testid="stDataFrame"],
[data-testid="stSidebar"],
[data-testid="stExpander"],
.stButton button,
.callout { box-shadow: none !important; }

/* ---------------- application header ---------------- */
.app-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
    margin-bottom: 16px;
}
.app-header-left { display: flex; align-items: baseline; gap: 12px; }
.app-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
    text-transform: uppercase;
    letter-spacing: 0.10em;
}
.app-subtitle {
    font-size: 11px;
    color: var(--dim);
    letter-spacing: 0.04em;
    border-left: 1px solid var(--border);
    padding-left: 12px;
}
.app-status {
    font-size: 11px;
    font-weight: 600;
    color: var(--green);
    letter-spacing: 0.10em;
    text-transform: uppercase;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 3px 8px;
}

/* ---------------- panels ---------------- */
.panel-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.09em;
    padding-bottom: 8px;
    margin: 0 0 12px;
    border-bottom: 1px solid var(--border);
}
.callout {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-left-width: 2px;
    border-radius: var(--radius);
    padding: 12px 14px;
    margin-bottom: 8px;
}
.callout-title {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-bottom: 6px;
}
.callout-body {
    font-size: 11px;
    color: var(--muted);
    line-height: 1.55;
    margin: 0;
}
.kv {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    font-size: 11px;
    color: var(--muted);
    padding: 4px 0;
}
.kv + .kv { border-top: 1px solid var(--border); }
.kv-label { letter-spacing: 0.02em; }
.kv-value { color: var(--text); font-weight: 600; font-variant-numeric: tabular-nums; }
.legend-note { font-size: 11px; color: var(--muted); line-height: 1.7; }
.legend-note p { margin: 0 0 4px; }
.legend-note b { color: var(--text); font-weight: 600; }

/* ---------------- metrics ---------------- */
[data-testid="stMetric"] {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 14px;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
    font-size: 11px !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
[data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    line-height: 1.2 !important;
    font-variant-numeric: tabular-nums;
}
[data-testid="stMetricDelta"] {
    font-size: 11px !important;
    font-weight: 500 !important;
    font-variant-numeric: tabular-nums;
}
/* Delta values are already signed; drop the decorative arrow glyphs */
[data-testid="stMetricDelta"] svg { display: none; }

/* ---------------- tabs ---------------- */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent;
    border-bottom: 1px solid var(--border);
    border-radius: 0;
    gap: 0;
    padding: 0;
    margin-bottom: 16px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border-radius: 0;
    color: var(--muted);
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 10px 16px;
    border-bottom: 2px solid transparent;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text); background-color: var(--surface); }
.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    background-color: transparent !important;
    border-bottom: 2px solid var(--blue) !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------------- sidebar ---------------- */
[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] h2 {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] label p {
    font-size: 11px !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    letter-spacing: 0.03em;
}
.sidebar-section {
    font-size: 10px;
    font-weight: 600;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}
.asset-row {
    font-size: 10px;
    font-weight: 600;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.10em;
}
.status-line {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--muted);
    border: 1px solid var(--border);
    border-left: 2px solid var(--green);
    border-radius: var(--radius);
    padding: 8px 10px;
}
.status-line b { color: var(--text); font-variant-numeric: tabular-nums; }

/* ---------------- inputs ---------------- */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"] {
    background-color: var(--bg) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-size: 12px !important;
    font-variant-numeric: tabular-nums;
}
[data-testid="stSidebar"] input:hover,
[data-testid="stSidebar"] [data-baseweb="input"]:hover,
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
    border-color: var(--border-strong) !important;
}
[data-testid="stSidebar"] input:focus,
[data-testid="stSidebar"] [data-baseweb="input"]:focus-within {
    border-color: var(--blue) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: var(--muted) !important; }
[data-testid="stSidebar"] .stNumberInput button {
    background-color: var(--surface-alt) !important;
    border-left: 1px solid var(--border) !important;
    color: var(--muted) !important;
}
[data-testid="stSidebar"] .stNumberInput button:hover { color: var(--text) !important; }

.stButton button {
    background-color: var(--blue) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--blue) !important;
    border-radius: var(--radius) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 9px 14px !important;
    transition: background-color 0.12s ease;
}
.stButton button:hover {
    background-color: #2F6FD1 !important;
    border-color: #2F6FD1 !important;
    transform: none !important;
}

[data-testid="stExpander"] {
    background-color: var(--surface-alt);
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stExpander"] summary {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stExpander"] summary:hover { color: var(--text) !important; }

/* ---------------- data display ---------------- */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    font-variant-numeric: tabular-nums;
}
[data-testid="stAlert"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-left: 2px solid var(--blue) !important;
    border-radius: var(--radius) !important;
    color: var(--muted) !important;
    font-size: 12px;
}
[data-testid="stImage"] img, [data-testid="stImageContainer"] img {
    border: 1px solid var(--border);
    border-radius: var(--radius);
}
hr, [data-testid="stDivider"] hr { border-color: var(--border); margin: 4px 0; }
.stSpinner > div { border-top-color: var(--blue) !important; }
</style>
""" % {
    "bg": BG,
    "surface": SURFACE,
    "surface_alt": SURFACE_ALT,
    "border": BORDER,
    "border_strong": BORDER_STRONG,
    "text": TEXT,
    "muted": TEXT_MUTED,
    "dim": TEXT_DIM,
    "green": GREEN,
    "red": RED,
    "blue": BLUE,
    "amber": AMBER,
    "radius": RADIUS,
    "gutter": GUTTER,
    "font": FONT_STACK,
}


def inject_css():
    """Install the global stylesheet. Call once, immediately after set_page_config."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def header(title, subtitle, status="Live"):
    """Render the application header bar."""
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-header-left">
                <div class="app-title">{title}</div>
                <div class="app-subtitle">{subtitle}</div>
            </div>
            <div class="app-status">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_title(text):
    """Render a dense uppercase section header with a hairline rule."""
    st.markdown(f'<div class="panel-title">{text}</div>', unsafe_allow_html=True)


def sidebar_section(text):
    st.markdown(f'<div class="sidebar-section">{text}</div>', unsafe_allow_html=True)


def callout(title, body=None, rows=None, tone="neutral"):
    """Flat panel with a 2px semantic accent on the leading edge.

    rows: iterable of (label, value) pairs rendered as dense key/value lines.
    """
    accent = tone_color(tone)
    parts = [
        f'<div class="callout" style="border-left-color:{accent};">',
        f'<div class="callout-title" style="color:{accent};">{title}</div>',
    ]
    if body:
        parts.append(f'<p class="callout-body">{body}</p>')
    for label, value in rows or []:
        parts.append(
            f'<div class="kv"><span class="kv-label">{label}</span>'
            f'<span class="kv-value">{value}</span></div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def status_line(label, value):
    st.markdown(
        f'<div class="status-line"><span>{label}</span><b>{value}</b></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Matplotlib styling
# --------------------------------------------------------------------------
def apply_chart_theme():
    """Apply the dark terminal chart style to matplotlib globally."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "figure.edgecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BORDER,
        "axes.linewidth": 0.8,
        "axes.labelcolor": TEXT_MUTED,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.titlecolor": TEXT,
        "axes.titlelocation": "left",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "text.color": TEXT_MUTED,
        "xtick.color": TEXT_DIM,
        "ytick.color": TEXT_DIM,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Roboto", "Segoe UI", "DejaVu Sans"],
        "font.size": 9,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.1,
        "lines.solid_capstyle": "butt",
        "patch.linewidth": 0,
    })


def figure(figsize, nrows=1, ncols=1, **kwargs):
    """Create a figure whose canvas matches the card surface."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    for ax in (axes.flatten() if hasattr(axes, "flatten") else [axes]):
        ax.set_facecolor(SURFACE)
    return fig, axes


def style_axes(ax, grid="y", bottom_spine=True):
    """Strip chart junk: no top/right spines, hairline single-axis grid only."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_visible(bottom_spine)
    ax.spines["bottom"].set_color(BORDER)
    ax.grid(False)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(colors=TEXT_DIM, labelsize=8, length=3, width=0.8)
    return ax


def currency_axis(ax, axis="y", decimals=0):
    fmt = plt.FuncFormatter(lambda v, _: f"£{v:,.{decimals}f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def percent_axis(ax, axis="y", decimals=1):
    fmt = plt.FuncFormatter(lambda v, _: f"{v:.{decimals}%}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def legend(ax, **kwargs):
    """Flat legend: no frame, muted labels, tight spacing."""
    opts = dict(frameon=False, fontsize=8, labelcolor=TEXT_MUTED,
                borderpad=0, handlelength=1.6, handletextpad=0.6,
                columnspacing=1.4)
    opts.update(kwargs)
    return ax.legend(**opts)


def legend_above(ax, ncol=1, **kwargs):
    """Legend parked in the title strip so it can never sit on top of data."""
    opts = dict(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=ncol)
    opts.update(kwargs)
    return legend(ax, **opts)


def correlation_cmap():
    """Diverging red -> surface -> green ramp built from the palette."""
    return LinearSegmentedColormap.from_list(
        "terminal_corr", [RED, RED_FILL, SURFACE, GREEN_FILL, GREEN]
    )
