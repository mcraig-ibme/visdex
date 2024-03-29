"""
visdex: Manhattan graph
"""
import logging
import numpy as np
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

import visdex.session

LOG = logging.getLogger(__name__)

# TODO: currently only allows int64 and float64
valid_manhattan_dtypes = [np.int64, np.float64]

def make_figure(dff, args_dict):
    if args_dict["base_variable"] is None or args_dict["base_variable"] == []:
        LOG.debug(f"return go.Figure()")
        raise PreventUpdate

    if args_dict["pvalue"] is None or args_dict["pvalue"] <= 0.0:
        LOG.debug(f"raise PreventUpdate")
        raise PreventUpdate

    # Load logs of all p-values
    logs = visdex.session.get().load("logs")

    # Select the column and row associated to this variable, and combine the two. Half of the values will be nans,
    # so we keep all the non-nans.
    take_non_nan = lambda s1, s2: s1 if not np.isnan(s1) else s2
    selected_logs = (
        logs.loc[:, args_dict["base_variable"]]
        .combine(logs.loc[args_dict["base_variable"], :], take_non_nan)
        .dropna()
    )

    transformed_corrected_ref_pval = calculate_transformed_corrected_pval(
        float(args_dict["pvalue"]), selected_logs
    )

    fig = go.Figure(
        go.Scatter(x=selected_logs.index, y=selected_logs.values, mode="markers"),
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
                x1=len(selected_logs) - 1,
            )
        ],
        annotations=[
            dict(
                x=0,
                y=transformed_corrected_ref_pval
                if args_dict["logscale"] != ["LOG"]
                else np.log10(transformed_corrected_ref_pval),
                xref="x",
                yref="y",
                text="{:f}".format(transformed_corrected_ref_pval),
                showarrow=True,
                arrowhead=7,
                ax=-50,
                ay=0,
            ),
        ],
        xaxis_title="variable",
        yaxis_title="-log10(p)",
        yaxis_type="log" if args_dict["logscale"] == ["LOG"] else None,
        title=f"Manhattan plot with base variable {args_dict['base_variable']} and p-value reference of {args_dict['pvalue']}",
    )
    return fig

def calculate_transformed_corrected_pval(ref_pval, logs):
    # Divide reference p-value by number of variable pairs to get corrected p-value
    corrected_ref_pval = ref_pval / (logs.notna().sum().sum())
    # Transform corrected p-value by -log10
    transformed_corrected_ref_pval = -np.log10(corrected_ref_pval)
    return transformed_corrected_ref_pval

