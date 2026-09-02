"""Single run detail page: metric charts, hyperparameters, hardware, notes."""

from __future__ import annotations

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from backend.config import settings
from dashboard import api
from dashboard.components.chart_theme import WARM_SEQUENCE, apply_theme, empty_note

dash.register_page(__name__, path_template="/runs/<run_id>", name="Run detail")

_STATUS_CLASS = {"running": "running", "completed": "completed", "failed": "failed"}


def _status_pill(status: str) -> html.Span:
    cls = _STATUS_CLASS.get(status, "")
    return html.Span(
        [html.Span(className="dot"), status], className=f"ft-pill {cls}"
    )


def layout(run_id: str | None = None, **_kwargs) -> html.Div:
    if not run_id:
        return html.Div("No run selected.", className="ft-muted")
    return html.Div(
        [
            dcc.Store(id="detail-run-id", data=run_id),
            dcc.Interval(id="detail-refresh", interval=settings.refresh_interval * 1000),
            html.Div(id="detail-header"),
            html.Div(
                dcc.Graph(id="detail-metrics-chart", config={"displayModeBar": False}),
                className="ft-panel",
                style={"marginTop": "1.6rem"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Hyperparameters", className="ft-section-label"),
                            html.Div(id="detail-hparams"),
                        ],
                        style={"flex": "1", "minWidth": "260px"},
                    ),
                    html.Div(
                        [
                            html.Div("Hardware", className="ft-section-label"),
                            html.Div(id="detail-hardware"),
                        ],
                        style={"flex": "1", "minWidth": "260px"},
                    ),
                ],
                style={"display": "flex", "gap": "2.4rem", "flexWrap": "wrap"},
            ),
            html.Div("Notes", className="ft-section-label"),
            dcc.Textarea(id="detail-notes", className="ft-notes"),
            html.Div(
                html.Button("Save notes", id="detail-save-notes", className="ft-btn"),
                style={"marginTop": "0.8rem"},
            ),
            html.Div(id="detail-notes-status", className="ft-muted", style={"marginTop": "0.5rem", "color": "#E6CE9B"}),
            html.Div(
                dcc.Link("← Back to runs", href="/", className="ft-back"),
                style={"marginTop": "2.4rem"},
            ),
        ]
    )


def _kv_table(data: dict) -> html.Div:
    if not data:
        return html.Div("None recorded.", className="ft-muted")
    return html.Table(
        html.Tbody(
            [html.Tr([html.Td(k), html.Td(str(v))]) for k, v in data.items()]
        ),
        className="ft-kv",
    )


@callback(
    Output("detail-header", "children"),
    Output("detail-metrics-chart", "figure"),
    Output("detail-hparams", "children"),
    Output("detail-hardware", "children"),
    Input("detail-run-id", "data"),
    Input("detail-refresh", "n_intervals"),
)
def _render_detail(run_id, _n):
    run = api.get_run(run_id)
    metrics = api.get_metrics(run_id)

    tags = run.get("tags") or []
    header = html.Div(
        [
            html.Span("Run detail", className="ft-eyebrow"),
            html.H1(run["name"], className="ft-page-title"),
            html.Div(
                [
                    _status_pill(run["status"]),
                    html.Div(
                        [html.Span("Base model", className="k"), html.Span(run.get("base_model") or "—", className="v")],
                        className="ft-meta",
                    ),
                    html.Div(
                        [html.Span("Dataset", className="k"), html.Span(run.get("dataset") or "—", className="v")],
                        className="ft-meta",
                    ),
                    html.Div(
                        [html.Span("Best eval loss", className="k"),
                         html.Span(
                             f"{run['best_eval_loss']:.4f}" if run.get("best_eval_loss") is not None else "—",
                             className="v",
                         )],
                        className="ft-meta",
                    ),
                ],
                className="ft-meta-row",
                style={"alignItems": "center"},
            ),
            html.Div(
                [html.Span(t, className="ft-tag") for t in tags] or html.Span("no tags", className="ft-muted"),
                style={"marginTop": "1rem"},
            ),
        ]
    )

    fig = go.Figure()
    by_name: dict[str, list[dict]] = {}
    for m in metrics:
        by_name.setdefault(m["name"], []).append(m)
    for i, (name, points) in enumerate(by_name.items()):
        xs = [p.get("step") for p in points]
        ys = [p.get("value") for p in points]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=name,
                line=dict(
                    color=WARM_SEQUENCE[i % len(WARM_SEQUENCE)],
                    width=2,
                    shape="spline",
                    smoothing=0.6,
                ),
            )
        )
    apply_theme(fig, x_title="STEP", y_title="VALUE")
    if not fig.data:
        empty_note(fig, "no metrics logged yet")

    return (
        header,
        fig,
        _kv_table(run.get("hyperparameters") or {}),
        _kv_table(run.get("hardware") or {}),
    )


@callback(
    Output("detail-notes", "value"),
    Input("detail-run-id", "data"),
)
def _load_notes(run_id):
    return api.get_run(run_id).get("notes") or ""


@callback(
    Output("detail-notes-status", "children"),
    Input("detail-save-notes", "n_clicks"),
    State("detail-run-id", "data"),
    State("detail-notes", "value"),
    prevent_initial_call=True,
)
def _save_notes(_clicks, run_id, notes):
    api.update_run(run_id, notes=notes or "")
    return "Notes saved."
