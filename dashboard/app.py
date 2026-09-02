"""Dash application factory.

The app is created with ``use_pages=True`` and mounted onto the FastAPI server
by ``backend/main.py``. Pages live in ``dashboard/pages``. Styling comes from
``dashboard/assets/theme.css`` (auto-loaded by Dash).
"""

from __future__ import annotations

import dash
from dash import Dash, dcc, html

_INDEX = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Fine-tune Tracker</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""


def _navbar() -> html.Div:
    return html.Div(
        className="ft-navbar",
        children=[
            dcc.Link(
                href="/",
                className="ft-brand",
                children=[
                    html.Span(className="ft-mark"),
                    html.Span("Fine-tune Tracker"),
                    html.Span("· runs & metrics", className="ft-brand-sub"),
                ],
            ),
            html.Div(
                className="ft-nav-links",
                children=[
                    dcc.Link("Runs", href="/"),
                    dcc.Link("Experiments", href="/experiments"),
                ],
            ),
        ],
    )


def create_dash_app() -> Dash:
    app = Dash(
        __name__,
        use_pages=True,
        pages_folder="pages",
        suppress_callback_exceptions=True,
        title="Fine-tune Tracker",
        index_string=_INDEX,
    )

    app.layout = html.Div(
        [
            _navbar(),
            html.Div(dash.page_container, className="ft-container"),
        ]
    )
    return app
