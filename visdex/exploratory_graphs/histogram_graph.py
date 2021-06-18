import logging
from dash.dependencies import Input, Output, State, MATCH
import plotly.graph_objects as go
from visdex.app import app, cache, all_components
from visdex.exploratory_graph_groups import update_graph_components

logging.getLogger(__name__)


@app.callback(
    [
        Output({"type": "div-histogram-" + component["id"], "index": MATCH}, "children")
        for component in all_components["histogram"]
    ],
    [Input("df-loaded-div", "children")],
    [State({"type": "div-histogram-base_variable", "index": MATCH}, "style")]
    + [
        State({"type": "histogram-" + component["id"], "index": MATCH}, prop)
        for component in all_components["histogram"]
        for prop in component
    ],
)
def update_histogram_components(df_loaded, style_dict, *args):
    logging.info(f"update_histogram_components")
    dff = cache.load("filtered")
    dd_options = [{"label": col, "value": col} for col in dff.columns]
    return update_graph_components(
        "histogram", all_components["histogram"], dd_options, args
    )


@app.callback(
    Output({"type": "gen-histogram-graph", "index": MATCH}, "figure"),
    [
        *(
            Input({"type": "histogram-" + component["id"], "index": MATCH}, "value")
            for component in all_components["histogram"]
        )
    ],
)
def make_histogram_figure(*args):
    logging.info(f"make_histogram_figure")
    keys = [component["id"] for component in all_components["histogram"]]

    args_dict = dict(zip(keys, args))
    dff = cache.load("filtered")

    # Return empty scatter if not enough options are selected, or the data is empty.
    if dff.columns.size == 0 or args_dict["base_variable"] is None:
        return go.Figure(go.Histogram())

    data_column = dff[args_dict["base_variable"]]

    fig = go.Figure(
        data=go.Histogram(
            x=data_column,
            xbins=dict(
                size=(data_column.max() - data_column.min()) / (args_dict["nbins"] - 1)
            ),
            autobinx=False,
        ),
    )

    fig.update_layout(
        yaxis_zeroline=False,
        xaxis_title=args_dict["base_variable"],
        yaxis_title="count",
        title=f'Histogram of {args_dict["base_variable"]} with '
        f"{args_dict['nbins']  if args_dict['nbins'] > 1 else 'automatically detected'} bins.",
    )

    return fig
