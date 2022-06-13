"""
visdex: Violin graph
"""
import logging
import plotly.graph_objects as go

from .common import common_define_cbs

LOG = logging.getLogger(__name__)

def define_cbs(app):
    common_define_cbs(app, "violin", make_violin_figure)

def make_violin_figure(dff, args_dict):
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
