import logging
from dash.dependencies import Input, Output, State, MATCH
import plotly.graph_objects as go
from visdex.app import app, all_components
from visdex.load_feather import load
from visdex.exploratory_graph_groups import update_graph_components

logging.getLogger(__name__)


@app.callback(
    [
        Output({"type": "div-bar-" + component["id"], "index": MATCH}, "children")
        for component in all_components["bar"]
    ],
    [Input("df-loaded-div", "children")],
    [State({"type": "div-bar-x", "index": MATCH}, "style")]
    + [
        State({"type": "bar-" + component["id"], "index": MATCH}, prop)
        for component in all_components["bar"]
        for prop in component
    ],
)
def update_bar_components(df_loaded, style_dict, *args):
    logging.info(f"update_bar_components")
    dff = load("filtered")
    dd_options = [{"label": col, "value": col} for col in dff.columns]
    return update_graph_components("bar", all_components["bar"], dd_options, args)


@app.callback(
    Output({"type": "gen-bar-graph", "index": MATCH}, "figure"),
    [
        *(
            Input({"type": "bar-" + component["id"], "index": MATCH}, "value")
            for component in all_components["bar"]
        )
    ],
)
def make_bar_figure(*args):
    logging.info(f"make_bar_figure")
    keys = [component["id"] for component in all_components["bar"]]

    args_dict = dict(zip(keys, args))
    dff = load("filtered")

    # Return empty scatter if not enough options are selected, or the data is empty.
    if dff.columns.size == 0 or args_dict["x"] is None:
        return go.Figure(go.Bar())

    fig = go.Figure()

    if args_dict["split_by"] is not None:
        # get all unique values in split_by column.
        # Filter by each of these, and add a go.Bar for each, and use these as names.
        split_by_names = sorted(dff[args_dict["split_by"]].dropna().unique())

        for name in split_by_names:
            count_by_value = dff[dff[args_dict["split_by"]] == name][
                args_dict["x"]
            ].value_counts()

            fig.add_trace(
                go.Bar(name=name, x=count_by_value.index, y=count_by_value.values)
            )
    else:

        count_by_value = dff[args_dict["x"]].value_counts()

        fig.add_trace(
            go.Bar(
                name=str(args_dict["x"]),
                x=count_by_value.index,
                y=count_by_value.values,
            )
        )
    fig.update_layout(coloraxis=dict(colorscale="Bluered_r"))

    return fig
