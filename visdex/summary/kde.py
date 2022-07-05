"""
visdex: Display summary KDEs
"""
import math

import numpy as np
import scipy.stats as stats

import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import html, dcc
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from visdex.common.timing import timing
import visdex.session
from visdex.common import Collapsible

class SummaryKdes(Collapsible):
    """
    """
    def __init__(self, app, id_prefix="kde-"):
        """
        :param app: Dash application
        """
        Collapsible.__init__(self, app, id_prefix, title="Per-variable Histograms and KDEs", children=[
            "Select (numerical) variables for KDE display",
            dcc.Dropdown(
                id=id_prefix+"dropdown",
                options=([]),
                multi=True,
            ),
            dcc.Loading(
                id=id_prefix+"loading",
                children=[dcc.Graph(id=id_prefix+"figure", figure=go.Figure())],
            ),
        ])

        self.register_cb(app, "update_dropdown", 
            [
                Output(id_prefix+"dropdown", "options"),
                Output(id_prefix+"dropdown", "value")
            ],
            [
                Input("filtered-loaded-div", "children")
            ],
            prevent_initial_call=True,
        )

        self.register_cb(app, "update_figure",
            Output(id_prefix+"figure", "figure"),
            [
                Input(id_prefix+"dropdown", "value"),
            ],
            [
                State("filtered-loaded-div", "children")
            ],
            prevent_initial_call=True,
        )

    @timing
    def update_dropdown(self, df_loaded):
        self.log.info(f"update_dropdown {df_loaded}")
        ds = visdex.session.get()
        dff = ds.load(visdex.session.FILTERED)

        options = [
            {"label": col, "value": col}
            for col in dff.columns
            if dff[col].dtype in [np.int64, np.float64]
        ]
        return options, []

    @timing
    def update_figure(self, dropdown_values, df_loaded):
        self.log.info(f"update_figure")
        ds = visdex.session.get()

        # Guard against the third argument being an empty list, as happens at first
        # invocation
        if df_loaded is False:
            return go.Figure(go.Scatter())

        dff = ds.load(visdex.session.FILTERED)
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
        fig.update_layout(height=100 + 200 * n_rows, showlegend=False)
        return fig
