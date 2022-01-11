"""
visdex: Summary manhattan plot
"""
import logging

import numpy as np

from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
from dash import html, dcc
import plotly.graph_objects as go

from visdex.cache import cache
from visdex.common import div_style
from visdex.timing import timing, start_timer, log_timing, print_timings
from visdex.exploratory_graphs.manhattan_graph import (
    calculate_transformed_corrected_pval,
)

LOG = logging.getLogger(__name__)

# TODO: currently only allows int64 and float64
valid_manhattan_dtypes = [np.int64, np.float64]

def get_layout(app):
    @app.callback(
        Output("manhattan-figure", "figure"),
        [
            Input("manhattan-pval-input", "value"),
            Input("manhattan-logscale-check", "value"),
            Input("df-filtered-loaded-div", "children"),
            Input("pval-loaded-div", "children"),
            Input("manhattan-active-check", "value"),
        ],
        prevent_initial_call=True,
    )
    @timing
    def plot_manhattan(pvalue, logscale, df_loaded, pval_loaded, manhattan_active):
        LOG.info(f"plot_manhattan")

        start_timer("plot_manhattan")
        if manhattan_active != ["manhattan-active"]:
            raise PreventUpdate
        if pvalue <= 0.0 or pvalue is None:
            raise PreventUpdate

        dff = cache.load("pval")

        log_timing("plot_manhattan", "plot_manhattan-load_pval")

        manhattan_variable = [
            col for col in dff.columns if dff[col].dtype in valid_manhattan_dtypes
        ]

        if not pval_loaded or manhattan_variable is None or manhattan_variable == []:
            return go.Figure()

        # Load logs and flattened logs from feather file.
        logs = cache.load("logs")
        flattened_logs = cache.load("flattened_logs")

        log_timing("plot_manhattan", "plot_manhattan-load_both_logs")

        transformed_corrected_ref_pval = calculate_transformed_corrected_pval(
            float(pvalue), logs
        )

        log_timing("plot_manhattan", "plot_manhattan-tranform")

        inf_replacement = 0
        if np.inf in flattened_logs.values:
            LOG.debug(f"Replacing np.inf in flattened logs")
            temp_vals = flattened_logs.replace([np.inf, -np.inf], np.nan).dropna()
            inf_replacement = 1.2 * np.max(np.flip(temp_vals.values))
            flattened_logs = flattened_logs.replace([np.inf, -np.inf], inf_replacement)

        log_timing("plot_manhattan", "plot_manhattan-load_cutoff")

        # Load cluster numbers to use for colouring
        cluster_df = cache.load("cluster")

        log_timing("plot_manhattan", "plot_manhattan-load_cluster")

        # Convert to colour array - set to the cluster number if the two variables are in
        # the same cluster, and set any other pairings to -1 (which will be coloured black)
        colors = [
            cluster_df["column_names"][item[0]]
            if cluster_df["column_names"][item[0]] == cluster_df["column_names"][item[1]]
            else -1
            for item in flattened_logs.index[::-1]
        ]

        log_timing("plot_manhattan", "plot_manhattan-calc_colors")

        max_cluster = max(cluster_df["column_names"])
        # Create graph, unless there's no data, in which case create a blank graph
        if len(flattened_logs) > 0:
            fig = go.Figure(
                go.Scatter(
                    x=[
                        [item[i] for item in flattened_logs.index[::-1]]
                        for i in range(0, 2)
                    ],
                    y=np.flip(flattened_logs.values),
                    mode="markers",
                    marker=dict(
                        color=colors,
                        colorscale=calculate_colorscale(max_cluster + 1),
                        colorbar=dict(
                            tickmode="array",
                            # Offset by 0.5 to centre the text within the boxes
                            tickvals=[i - 0.5 for i in range(max_cluster + 2)],
                            # The bottom of the colorbar is black for "cross-cluster pair"s
                            ticktext=["cross-cluster pair"]
                            + [str(i) for i in range(max_cluster + 2)],
                            title="Cluster",
                        ),
                        cmax=max_cluster + 1,
                        # Minimum is -1 to represent the points of variables not in the
                        # same cluster
                        cmin=-1,
                        showscale=True,
                    ),
                ),
            )

            fig.update_layout(
                shapes=[
                    dict(
                        type="line",
                        yref="y",
                        y0=transformed_corrected_ref_pval,
                        y1=transformed_corrected_ref_pval,
                        xref="x",
                        x0=0,
                        x1=len(flattened_logs) - 1,
                    )
                ],
                annotations=[
                    # This annotation prints the transformed pvalue
                    dict(
                        x=0,
                        y=transformed_corrected_ref_pval
                        if logscale != ["LOG"]
                        else np.log10(transformed_corrected_ref_pval),
                        xref="x",
                        yref="y",
                        text="{:f}".format(transformed_corrected_ref_pval),
                        showarrow=True,
                        arrowhead=7,
                        ax=-50,
                        ay=0,
                    ),
                    # This annotation above the top of the graph is only shown if any inf
                    # values have been replaced. It highlights what value has been used to
                    # replace them (1.2*max)
                    dict(
                        x=0,
                        y=1.1,
                        xref="paper",
                        yref="paper",
                        text="Infinite values (corresponding to a 0 p-value after "
                        "numerical errors) have been replaced with {:f}".format(
                            inf_replacement
                        )
                        if inf_replacement != 0
                        else "",
                        showarrow=False,
                    ),
                ],
                xaxis_title="variable",
                yaxis_title="-log10(p)",
                yaxis_type="log" if logscale == ["LOG"] else None,
            )
        else:
            fig = go.Figure()

        log_timing("plot_manhattan", "plot_manhattan-figure", restart=False)

        print_timings()
        return fig

    return html.Div(children=[
        html.H2("Manhattan Plot", style=div_style),
        dcc.Loading(
            id="loading-manhattan-figure",
            children=[
                html.Div(
                    [
                        dcc.Checklist(
                            id="manhattan-active-check",
                            options=[
                                {
                                    "label": " Plot Manhattan",
                                    "value": "manhattan-active",
                                }
                            ],
                            value=[],
                            style={"display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        "p-value:  ",
                        dcc.Input(
                            id="manhattan-pval-input",
                            type="number",
                            value=0.05,
                            step=0.0001,
                            debounce=True,
                            style={"display": "inline-block"},
                        ),
                    ],
                    style=div_style,
                ),
                html.Div(
                    [
                        dcc.Checklist(
                            id="manhattan-logscale-check",
                            options=[
                                {"label": "  logscale y-axis", "value": "LOG"}
                            ],
                            value=[],
                            style={"display": "inline-block"},
                        ),
                    ],
                    style=div_style,
                ),
                dcc.Graph(id="manhattan-figure", figure=go.Figure()),
            ],
        ),
    ])

def calculate_colorscale(n_values):
    """
    Split the colorscale into n_values + 1 values. The first is black for
    points representing pairs of variables not in the same cluster.
    :param n_values:
    :return:
    """
    # First 2 entries define a black colour for the "none" category
    colorscale = [[0, "rgb(0,0,0)"], [1 / (n_values + 1), "rgb(0,0,0)"]]

    plotly_colorscale = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]

    # Add colours from plotly_colorscale as required.
    for i in range(1, n_values + 1):
        colorscale.append(
            [i / (n_values + 1), plotly_colorscale[(i - 1) % len(plotly_colorscale)]]
        )
        colorscale.append(
            [
                (i + 1) / (n_values + 1),
                plotly_colorscale[(i - 1) % len(plotly_colorscale)],
            ]
        )

    return colorscale
