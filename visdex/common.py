"""
visdex: Common layout definitions
"""
import logging

from dash import html, dcc

# Default style constants
standard_margin_left = "10px"
default_marker_color = "crimson"

# Vertically stacked page elements
vstack = {
    "margin-left": standard_margin_left
}

# Page elements that can stack horizontally
hstack = {
    "margin-left": standard_margin_left,
    "display": "inline-block",
}

# Style for exploratory graphs
plot_style = {
    "width": "13%",
    "display": "inline-block",
    "verticalAlign": "middle",
    "margin-right": "2em",
}

class Component(html.Div):
    """
    Base class for layout/callback component

    Not sure yet if this is the best way to modularize - currently used only for preview table
    """

    def __init__(self, app, id_prefix, *args, **kwargs):
        self.app = app
        self.id_prefix = id_prefix
        self.log = logging.getLogger(type(self).__name__)
        html.Div.__init__(self, *args, **kwargs)

    def register_cb(self, app, name, *args, **kwargs):
        @app.callback(*args, **kwargs)
        def _cb(*cb_args, **cb_kwargs):
            return getattr(self, name)(*cb_args, **cb_kwargs)

