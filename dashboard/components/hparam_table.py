"""Hyperparameter diff table across selected runs."""

from __future__ import annotations

from dash import html


def build_hparam_table(runs: list[dict]) -> html.Div:
    """Build a diff-style table: rows = hyperparameters, columns = runs.

    Rows where values differ across runs are highlighted.
    """
    if not runs:
        return html.Div(
            "Select runs to compare hyperparameters.", className="ft-muted"
        )

    all_keys: list[str] = []
    for run in runs:
        for key in (run.get("hyperparameters") or {}):
            if key not in all_keys:
                all_keys.append(key)

    if not all_keys:
        return html.Div(
            "No hyperparameters logged for the selected runs.", className="ft-muted"
        )

    header = html.Thead(
        html.Tr(
            [html.Th("Parameter")]
            + [html.Th(run.get("name", run["id"][:8])) for run in runs]
        )
    )

    body_rows = []
    for key in all_keys:
        values = [(run.get("hyperparameters") or {}).get(key) for run in runs]
        differs = len({repr(v) for v in values}) > 1
        body_rows.append(
            html.Tr(
                [html.Td(key)]
                + [html.Td("—" if v is None else str(v)) for v in values],
                className="diff" if differs else "",
            )
        )

    return html.Table(
        [header, html.Tbody(body_rows)], className="ft-hparam"
    )
