"""Shared dark 'analog instrument' styling for Plotly figures.

Warm charcoal paper, burnt-orange hero, clay/gold/amber supporting tones.
No blue / purple / green.
"""

from __future__ import annotations

import plotly.graph_objects as go

# Ordered warm palette for traces. Hero ember first, then supporting tones.
WARM_SEQUENCE = [
    "#E07A3C",  # ember (burnt orange)
    "#E6CE9B",  # warm gold / cream
    "#C0503C",  # terracotta clay
    "#E8A23C",  # amber
    "#A6795A",  # taupe brown
    "#D98E6A",  # soft clay
    "#C9A24B",  # brass
    "#8B5E34",  # deep umber
]

_PAPER = "rgba(0,0,0,0)"
_GRID = "rgba(58,47,38,0.55)"
_ZERO = "rgba(58,47,38,0.9)"
_TEXT = "#C9BBAA"
_TITLE = "#8B7C6C"

_FONT = "Space Grotesk, -apple-system, sans-serif"
_MONO = "JetBrains Mono, monospace"


def apply_theme(fig: go.Figure, x_title: str = "", y_title: str = "") -> go.Figure:
    """Apply the shared dark theme to a figure in place and return it."""
    fig.update_layout(
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PAPER,
        font=dict(family=_FONT, color=_TEXT, size=12),
        margin=dict(l=52, r=24, t=18, b=46),
        height=430,
        colorway=WARM_SEQUENCE,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#201a15",
            bordercolor="#3a2f26",
            font=dict(family=_MONO, color="#F2EBE1", size=11),
        ),
        legend=dict(
            bgcolor=_PAPER,
            font=dict(family=_MONO, size=10, color=_TEXT),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    axis_common = dict(
        gridcolor=_GRID,
        zerolinecolor=_ZERO,
        linecolor="#3a2f26",
        tickfont=dict(family=_MONO, size=10, color=_TITLE),
        title_font=dict(family=_MONO, size=11, color=_TITLE),
    )
    fig.update_xaxes(title_text=x_title, **axis_common)
    fig.update_yaxes(title_text=y_title, **axis_common)
    return fig


def empty_note(fig: go.Figure, text: str) -> None:
    """Add a centered muted annotation for empty states."""
    fig.add_annotation(
        text=text,
        showarrow=False,
        font=dict(family=_MONO, size=13, color="#8B7C6C"),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
