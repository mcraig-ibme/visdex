"""
visdex: Common layout definitions
"""
import logging

from dash import html, dcc

# Default style constants
standard_margin_left = "10px"
default_marker_color = "crimson"
div_style = {
    "margin-left": standard_margin_left
}
hstack = {
    "display": "inline-block",
}
style_dict = {
    "width": "13%",
    "display": "inline-block",
    "verticalAlign": "middle",
    "margin-right": "2em",
}
GLOBAL_WIDTH = "100%"
TABLE_WIDTH = "95%"

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

# Definitions of supported plot types
# All components should contain 'component_type', 'id', and 'label' as a minimum
all_components = dict(
    scatter=[
        {
            "component_type": dcc.Dropdown,
            "id": "x",
            "label": "x",
            "value": None,
            "multi": False,
            "options": [],
        },
        {
            "component_type": dcc.Dropdown,
            "id": "y",
            "label": "y",
            "value": None,
            "multi": False,
            "options": [],
        },
        {
            "component_type": dcc.Dropdown,
            "id": "color",
            "label": "color",
            "value": None,
            "multi": False,
            "options": [],
        },
        {
            "component_type": dcc.Dropdown,
            "id": "size",
            "label": "size",
            "value": None,
            "multi": False,
            "options": [],
        },
        {
            "component_type": dcc.Dropdown,
            "id": "facet_col",
            "label": "split horizontally",
            "value": None,
            "multi": False,
            "options": [],
        },
        {
            "component_type": dcc.Dropdown,
            "id": "facet_row",
            "label": "split vertically",
            "value": None,
            "multi": False,
            "options": [],
        },
        {
            "component_type": dcc.Input,
            "id": "regression",
            "label": "regression degree",
            "value": None,
            "type": "number",
            "min": 0,
            "step": 1,
        },
    ],
    bar=[
        {
            "component_type": dcc.Dropdown,
            "id": "x",
            "label": "x",
            "value": None,
            "options": [],
            "multi": False,
        },
        {
            "component_type": dcc.Dropdown,
            "id": "split_by",
            "label": "split by",
            "value": None,
            "options": [],
            "multi": False,
        },
    ],
    manhattan=[
        {
            "component_type": dcc.Dropdown,
            "id": "base_variable",
            "label": "base variable",
            "value": None,
            "options": [],
            "multi": False,
        },
        {
            "component_type": dcc.Input,
            "id": "pvalue",
            "label": "p-value",
            "value": 0.05,
            "type": "number",
            "min": 0,
            "step": 0.001,
        },
        {
            "component_type": dcc.Checklist,
            "id": "logscale",
            "label": "logscale",
            "options": [{"label": "", "value": "LOG"}],
            "value": ["LOG"],
        },
    ],
    violin=[
        {
            "component_type": dcc.Dropdown,
            "id": "base_variable",
            "label": "base variable",
            "value": None,
            "options": [],
            "multi": False,
        },
    ],
    histogram=[
        {
            "component_type": dcc.Dropdown,
            "id": "base_variable",
            "label": "base variable",
            "value": None,
            "options": [],
            "multi": False,
        },
        {
            "component_type": dcc.Input,
            "id": "nbins",
            "label": "n bins (1=auto)",
            "value": 1,
            "type": "number",
            "min": 1,
            "step": 1,
        },
    ],
)
