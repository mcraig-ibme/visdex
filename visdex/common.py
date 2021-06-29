"""
Common layout definitions
"""
import logging

import dash_html_components as html
import dash_core_components as dcc

# Default style constants
standard_margin_left = "10px"
default_marker_color = "crimson"
div_style = {
    "margin-left": standard_margin_left
}
style_dict = {
    "width": "13%",
    "display": "inline-block",
    "verticalAlign": "middle",
    "margin-right": "2em",
}
GLOBAL_WIDTH = "100%"
TABLE_WIDTH = "95%"
HEADER_IMAGE = "/assets/UoN_Primary_Logo_RGB.png"

class Component(html.Div):
    """
    Base class for layout/callback component
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

def create_header(title):
    """
    Create header for UoN DRS application
    """
    return html.Div(
        id="app-header",
        children=[
            html.A(
                id="drs-link",
                children=[
                    html.H1("DRS |"),
                ],
                href="https://digitalresearch.nottingham.ac.uk/",
                style={
                    "display": "inline-block",
                    "margin-left": "10px",
                    "margin-right": "10px",
                    "text-decoration": "none",
                    "align": "center",
                },
            ),
            html.H2(
                title,
                style={
                    "display": "inline-block",
                },
            ),
            html.A(
                id="uni-link",
                children=[
                    html.H1("UoN"),
                ],
                href="https://www.nottingham.ac.uk/",
                style={
                    "display": "inline-block",
                    "float": "right",
                    "margin-right": "10px",
                    "text-decoration": "none",
                    "color": "#FFFFFF",
                },
            ),
        ],
        style={"width": "100%", "display": "inline-block", "color": "white"},
    )
