import logging
import numpy as np
from dash.dependencies import Input, Output, State, MATCH
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from visdex.app import app, cache, all_components
from visdex.exploratory_graph_groups import update_graph_components

logging.getLogger(__name__)

# TODO: currently only allows int64 and float64
valid_manhattan_dtypes = [np.int64, np.float64]


@app.callback(
    [
        Output({"type": "div-manhattan-" + component["id"], "index": MATCH}, "children")
        for component in all_components["manhattan"]
    ],
    [Input("df-loaded-div", "children")],
    [State({"type": "div-manhattan-base_variable", "index": MATCH}, "style")]
    + [
        State({"type": "manhattan-" + component["id"], "index": MATCH}, prop)
        for component in all_components["manhattan"]
        for prop in component
    ],
)
def update_manhattan_components(df_loaded, style_dict, *args):
    logging.info("update_manhattan_components")
    dff = cache.load("df")
    # Only allow user to select columns that have data type that is valid for correlation
    dd_options = [
        {"label": col, "value": col}
        for col in dff.columns
        if dff[col].dtype in valid_manhattan_dtypes
    ]
    return update_graph_components(
        "manhattan", all_components["manhattan"], dd_options, args
    )


def calculate_transformed_corrected_pval(ref_pval, logs):
    # Divide reference p-value by number of variable pairs to get corrected p-value
    corrected_ref_pval = ref_pval / (logs.notna().sum().sum())
    # Transform corrected p-value by -log10
    transformed_corrected_ref_pval = -np.log10(corrected_ref_pval)
    return transformed_corrected_ref_pval


@app.callback(
    Output({"type": "gen-manhattan-graph", "index": MATCH}, "figure"),
    [
        *(
            Input({"type": "manhattan-" + component["id"], "index": MATCH}, "value")
            for component in all_components["manhattan"]
        )
    ],
)
def make_manhattan_figure(*args):
    args_string = [*args]
    logging.info(f"make_manhattan_figure {args_string}")
    # Generate the list of argument names based on the input order
    keys = [component["id"] for component in all_components["manhattan"]]

    # Convert inputs to a dict called 'args_dict'
    args_dict = dict(zip(keys, args))
    if args_dict["base_variable"] is None or args_dict["base_variable"] == []:
        logging.debug(f"return go.Figure()")
        raise PreventUpdate

    if args_dict["pvalue"] is None or args_dict["pvalue"] <= 0.0:
        logging.debug(f"raise PreventUpdate")
        raise PreventUpdate

    # Load logs of all p-values
    logs = cache.load("logs")

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
