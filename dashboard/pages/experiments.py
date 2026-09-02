"""Experiments overview page: list experiments with run count and best metric."""

from __future__ import annotations

import dash
from dash import Input, Output, callback, dcc, html

from backend.config import settings
from dashboard import api

dash.register_page(__name__, path="/experiments", name="Experiments")


def layout() -> html.Div:
    return html.Div(
        [
            html.Span("Grouped Work", className="ft-eyebrow"),
            html.H1("Experiments", className="ft-page-title"),
            html.P(
                "Each experiment collects related runs. Best eval loss is the "
                "lowest recorded across all its runs.",
                className="ft-lede",
            ),
            dcc.Interval(id="exp-refresh", interval=settings.refresh_interval * 1000),
            html.Div(id="experiments-table"),
        ]
    )


@callback(
    Output("experiments-table", "children"),
    Input("exp-refresh", "n_intervals"),
)
def _render_experiments(_n):
    experiments = api.list_experiments()
    if not experiments:
        return html.Div("No experiments yet.", className="ft-muted")

    header = html.Thead(
        html.Tr(
            [
                html.Th("Experiment"),
                html.Th("Runs"),
                html.Th("Best eval loss"),
                html.Th("Created"),
            ]
        )
    )
    rows = []
    for exp in experiments:
        best = exp.get("best_eval_loss")
        created = (exp.get("created_at") or "").replace("T", " ")[:19]
        rows.append(
            html.Tr(
                [
                    html.Td(exp["name"]),
                    html.Td(str(exp.get("run_count", 0))),
                    html.Td(f"{best:.4f}" if best is not None else "—"),
                    html.Td(created),
                ]
            )
        )
    return html.Table([header, html.Tbody(rows)], className="ft-hparam")
