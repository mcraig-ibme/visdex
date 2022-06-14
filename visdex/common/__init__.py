"""
visdex: Common layout definitions
"""
import logging

from dash import html, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

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

class Collapsible(Component):
    """
    Component which has an expand/collapse button
    """
    def __init__(self, app, id_prefix, title, is_open=False, *args, **kwargs):
        """
        :param app: Dash application
        """
        children=[
            html.Div(children=[
                dbc.Button(
                    "-" if is_open else "+",
                    id=id_prefix+"collapse-button",
                    style={
                        "display": "inline-block",
                        "margin-left": "10px",
                        "width": "40px",
                        "vertical-align" : "middle",
                    },
                ),
                html.H3(children=title,
                    style={
                        "display": "inline-block",
                        "vertical-align" : "middle",
                    }
                ),
            ]),
            dbc.Collapse(
                id=id_prefix+"collapse",
                style=vstack,
                children=kwargs.pop("children", []),
                is_open=is_open,                
            ),
            html.Div(id=id_prefix+"force-collapse", style={"display": "none"}, children=[]),
        ]

        Component.__init__(self, app, id_prefix, children=children, *args, **kwargs)

        self.register_cb(app, "toggle_collapse", 
            [
                Output(id_prefix+"collapse", "is_open"),
                Output(id_prefix+"collapse-button", "children"),
            ],
            [
                Input(id_prefix+"collapse-button", "n_clicks"),
                Input(id_prefix+"force-collapse", "children")
            ],
            [State(id_prefix+"collapse", "is_open")],
            prevent_initial_call=True,
        )

    def toggle_collapse(self, n_clicks, force_collapse, is_open):
        """
        Handle click on the expand/collapse button
        """
        self.log.info(f"toggle_collapse {n_clicks} {force_collapse} {is_open}")
        which_input = callback_context.triggered[0]['prop_id'].split('.')[0]
        if which_input == self.id_prefix+"force-collapse":
            if force_collapse:
                return False, "+"
            else:
                return True, "-"
        elif n_clicks:
            return not is_open, "+" if is_open else "-"
        else:
            return is_open, "-"
