"""Runs list page: table + multi-run compare panel with loss curves and hparam diff."""

from __future__ import annotations

import dash
from dash import Input, Output, callback, dash_table, dcc, html

from backend.config import settings
from dashboard import api
from dashboard.components.hparam_table import build_hparam_table
from dashboard.components.loss_chart import build_loss_figure

dash.register_page(__name__, path="/", name="Runs")

# Unicode dot + label rendered as markdown so we can color per status in-cell.
_STATUS_MD = {
    "running": "● running",
    "completed": "◆ completed",
    "failed": "▲ failed",
}


def _status_cell(status: str) -> str:
    return _STATUS_MD.get(status, status)


def layout() -> html.Div:
    return html.Div(
        [
            html.Span("Training Registry", className="ft-eyebrow"),
            html.H1("Fine-tuning Runs", className="ft-page-title"),
            html.P(
                "Every logged run, side by side. Select rows to overlay loss "
                "curves and diff their hyperparameters.",
                className="ft-lede",
            ),
            dcc.Interval(id="runs-refresh", interval=settings.refresh_interval * 1000),
            html.Div(
                dash_table.DataTable(
                    id="runs-table",
                    columns=[
                        {"name": "Run", "id": "name"},
                        {"name": "Status", "id": "status", "presentation": "markdown"},
                        {"name": "Base model", "id": "base_model"},
                        {"name": "Duration (s)", "id": "duration_s"},
                        {"name": "Best eval loss", "id": "best_eval_loss"},
                        {"name": "ID", "id": "id"},
                    ],
                    data=[],
                    row_selectable="multi",
                    sort_action="native",
                    filter_action="native",
                    page_size=15,
                    markdown_options={"html": False},
                    style_as_list_view=True,
                    style_cell={
                        "textAlign": "left",
                        "backgroundColor": "transparent",
                        "border": "none",
                    },
                    style_data_conditional=[
                        {
                            "if": {
                                "filter_query": '{status} contains "running"',
                                "column_id": "status",
                            },
                            "color": "#E8A23C",
                        },
                        {
                            "if": {
                                "filter_query": '{status} contains "completed"',
                                "column_id": "status",
                            },
                            "color": "#E6CE9B",
                        },
                        {
                            "if": {
                                "filter_query": '{status} contains "failed"',
                                "column_id": "status",
                            },
                            "color": "#C0503C",
                        },
                        {
                            "if": {"column_id": ["duration_s", "best_eval_loss", "id"]},
                            "fontFamily": "JetBrains Mono, monospace",
                            "color": "#C9BBAA",
                        },
                    ],
                ),
                className="ft-table",
            ),
            html.Div(id="runs-empty-message"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Comparison", className="ft-eyebrow"),
                            html.Div(
                                [
                                    html.Span("Loss overlay", className="ft-toggle-label"),
                                    dcc.RadioItems(
                                        id="compare-xaxis",
                                        options=[
                                            {"label": "Step", "value": "step"},
                                            {"label": "Epoch", "value": "epoch"},
                                        ],
                                        value="step",
                                        inline=True,
                                    ),
                                ],
                                className="ft-toggle",
                                style={"marginBottom": "1rem"},
                            ),
                            dcc.Graph(
                                id="compare-loss-chart",
                                config={"displayModeBar": False},
                            ),
                        ],
                        className="ft-panel",
                    ),
                    html.Div("Hyperparameters", className="ft-section-label"),
                    html.Div(id="compare-hparam-table"),
                ],
                style={"marginTop": "2.4rem"},
            ),
        ]
    )


@callback(
    Output("runs-table", "data"),
    Output("runs-empty-message", "children"),
    Input("runs-refresh", "n_intervals"),
)
def _refresh_runs_data(_n):
    """Update only the table's data on each interval, preserving selection."""
    runs = api.list_runs()
    if not runs:
        return [], html.Div(
            "No runs yet. Log one with the SDK to get started.",
            className="ft-muted",
            style={"padding": "1.5rem 0"},
        )

    rows = []
    for r in runs:
        duration = r.get("duration_seconds")
        rows.append(
            {
                "name": r["name"],
                "status": _status_cell(r["status"]),
                "base_model": r.get("base_model") or "—",
                "duration_s": round(duration, 1) if duration is not None else "—",
                "best_eval_loss": (
                    round(r["best_eval_loss"], 4)
                    if r.get("best_eval_loss") is not None
                    else "—"
                ),
                "id": r["id"],
            }
        )
    return rows, ""


@callback(
    Output("compare-loss-chart", "figure"),
    Output("compare-hparam-table", "children"),
    Input("runs-table", "selected_rows"),
    Input("runs-table", "data"),
    Input("compare-xaxis", "value"),
    prevent_initial_call=True,
)
def _render_compare(selected_rows, data, x_axis):
    if not selected_rows or not data:
        return build_loss_figure({"runs": {}}, {}, x_axis), build_hparam_table([])

    run_ids = [data[i]["id"] for i in selected_rows]
    labels = {data[i]["id"]: data[i]["name"] for i in selected_rows}
    compare_data = api.compare(run_ids)
    runs = [api.get_run(rid) for rid in run_ids]
    return (
        build_loss_figure(compare_data, labels, x_axis),
        build_hparam_table(runs),
    )
