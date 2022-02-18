import logging
import math

import numpy as np
import scipy.stats as stats

from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from dash import html, dcc
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from visdex.timing import timing
from visdex.data.cache import get_cache
from visdex.common import vstack

LOG = logging.getLogger(__name__)

def get_layout(app):
    @app.callback(
        Output("kde-figure", "figure"),
        [Input("heatmap-dropdown", "value"), Input("kde-checkbox", "value")],
        [State("df-loaded-div", "children")],
        prevent_initial_call=True,
    )
    @timing
    def update_summary_kde(dropdown_values, kde_active, df_loaded):
        LOG.info(f"update_summary_kde")
        cache = get_cache()
        if kde_active != ["kde-active"]:
            raise PreventUpdate

        # Guard against the third argument being an empty list, as happens at first
        # invocation
        if df_loaded is False:
            return go.Figure(go.Scatter())

        dff = cache.load("filtered")

        n_variables = len(dropdown_values) if dropdown_values is not None else 0

        # Return early if no variables are selected
        if n_variables == 0:
            return go.Figure(go.Scatter())

        # Use a maximum of 5 columns
        n_cols = min(5, math.ceil(math.sqrt(n_variables)))
        n_rows = math.ceil(n_variables / n_cols)
        fig = make_subplots(n_rows, n_cols, subplot_titles=dropdown_values)

        # For each column, calculate its KDE and then plot that over the histogram.
        for i in range(n_rows):
            for j in range(n_cols):
                if i * n_cols + j < n_variables:
                    col_name = dropdown_values[i * n_cols + j]
                    this_col = dff[col_name]
                    col_min = this_col.min()
                    col_max = this_col.max()
                    data_range = col_max - col_min
                    # Guard against a singular matrix in the KDE when a column
                    # contains only a single value
                    if data_range > 0:

                        # Generate KDE kernel
                        kernel = stats.gaussian_kde(this_col.dropna())
                        pad = 0.1
                        # Generate linspace
                        x = np.linspace(
                            col_min - pad * data_range,
                            col_max + pad * data_range,
                            num=21,
                        )
                        # Sample kernel
                        y = kernel(x)
                        # Plot KDE line graph using sampled data
                        fig.add_trace(go.Scatter(x=x, y=y, name=col_name), i + 1, j + 1)
                    # Plot normalised histogram of data, regardless of KDE
                    # completion or not
                    fig.add_trace(
                        go.Histogram(
                            x=this_col,
                            name=col_name,
                            histnorm="probability density",
                        ),
                        i + 1,
                        j + 1,
                    )
        fig.update_layout(height=200 * n_rows, showlegend=False)
        return fig

    return html.Div(children=[
        html.H3(children="Per-variable Histograms and KDEs", style=vstack),
        dcc.Checklist(
            "kde-checkbox",
            options=[{"label": " Run KDE analysis", "value": "kde-active"}],
            value=[],
            style=vstack,
        ),
        dcc.Loading(
            id="loading-kde-figure",
            children=[dcc.Graph(id="kde-figure", figure=go.Figure())],
        ),
    ])                
