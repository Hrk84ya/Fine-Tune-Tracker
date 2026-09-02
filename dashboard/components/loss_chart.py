"""Reusable multi-run loss curve chart."""

from __future__ import annotations

import plotly.graph_objects as go

from dashboard.components.chart_theme import (
    WARM_SEQUENCE,
    apply_theme,
    empty_note,
)

# Metric names rendered on the loss chart.
LOSS_METRICS = ("train_loss", "eval_loss", "val_loss")

# train solid, eval/val dashed — so a run's pair shares a hue but reads apart.
_DASH = {"train_loss": "solid", "eval_loss": "dot", "val_loss": "dash"}


def build_loss_figure(
    compare_data: dict,
    run_labels: dict[str, str],
    x_axis: str = "step",
) -> go.Figure:
    """Build an overlay figure of loss metrics across runs.

    Args:
        compare_data: payload from GET /api/compare -> {"runs": {run_id: {name: [pts]}}}
        run_labels: run_id -> human-readable label for the legend
        x_axis: "step" or "epoch"
    """
    fig = go.Figure()
    runs = compare_data.get("runs", {})
    for run_index, (run_id, by_name) in enumerate(runs.items()):
        label = run_labels.get(run_id, run_id[:8])
        hue = WARM_SEQUENCE[run_index % len(WARM_SEQUENCE)]
        for metric_name, points in by_name.items():
            if metric_name not in LOSS_METRICS:
                continue
            xs = [p.get(x_axis) for p in points]
            ys = [p.get("value") for p in points]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name=f"{label} · {metric_name}",
                    line=dict(
                        color=hue,
                        width=2,
                        dash=_DASH.get(metric_name, "solid"),
                        shape="spline",
                        smoothing=0.6,
                    ),
                )
            )
    apply_theme(fig, x_title=x_axis.upper(), y_title="LOSS")
    if not fig.data:
        empty_note(fig, "select runs above to compare")
    return fig
