"""
visdex: Histogram
"""
import logging

import plotly.graph_objects as go

from visdex.data import data_store
from .common import common_define_cbs

LOG = logging.getLogger(__name__)

def define_cbs(app):
    common_define_cbs(app, "histogram", make_histogram_figure)
    
def make_histogram_figure(dff, args_dict):
    # Return empty plot if not enough options are selected, or the data is empty.
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
