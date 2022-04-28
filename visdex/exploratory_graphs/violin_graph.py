import logging
from dash.dependencies import Input, Output, State, MATCH
import plotly.graph_objects as go
from visdex.data import data_store

from .common import all_components, update_graph_components

LOG = logging.getLogger(__name__)

def define_cbs(app):

    @app.callback(
        [
            Output({"type": "div-violin-" + component["id"], "index": MATCH}, "children")
            for component in all_components["violin"]
        ],
        [Input("filtered-loaded-div", "children")],
        [State({"type": "div-violin-base_variable", "index": MATCH}, "style")]
        + [
            State({"type": "violin-" + component["id"], "index": MATCH}, prop)
            for component in all_components["violin"]
            for prop in component
        ],
    )
    def update_violin_components(df_loaded, style_dict, *args):
        LOG.info(f"update_violin_components")
        ds = data_store.get()
        dff = ds.load(data_store.FILTERED)
        dd_options = [{"label": col, "value": col} for col in dff.columns]
        return update_graph_components("violin", all_components["violin"], dd_options, args)


    @app.callback(
        Output({"type": "gen-violin-graph", "index": MATCH}, "figure"),
        [
            *(
                Input({"type": "violin-" + component["id"], "index": MATCH}, "value")
                for component in all_components["violin"]
            )
        ],
    )
    def make_violin_figure(*args):
        LOG.info(f"make_violin_figure")
        ds = data_store.get()
        keys = [component["id"] for component in all_components["violin"]]

        args_dict = dict(zip(keys, args))
        dff = ds.load(data_store.FILTERED)

        # Return empty scatter if not enough options are selected, or the data is empty.
        if dff.columns.size == 0 or args_dict["base_variable"] is None:
            return go.Figure(go.Violin())

        fig = go.Figure(
            data=go.Violin(
                y=dff[args_dict["base_variable"]],
                box_visible=True,
                line_color="black",
                meanline_visible=True,
                fillcolor="lightseagreen",
                opacity=0.6,
                x0=args_dict["base_variable"],
            )
        )
        fig.update_layout(yaxis_zeroline=False)

        return fig
