"""
visdex: Bar graph
"""
import logging
import plotly.graph_objects as go

import visdex.session
from .common import common_define_cbs

LOG = logging.getLogger(__name__)

def define_cbs(app):
    common_define_cbs(app, "bar", make_bar_figure)

def make_bar_figure(dff, args_dict):
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
