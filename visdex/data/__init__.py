"""
visdex: Data selection

These components handle the loading of the main data
"""
from dash import html, dcc
from dash.dependencies import Input, Output

from visdex.common import Component
from . import data_store, user_upload, std_data

class DataSelection(Component):
    def __init__(self, app):
        Component.__init__(self, app, id_prefix="data-", children=[
            html.H2(children="Data selection"),
            html.Div(
                dcc.Dropdown(
                    id="dataset-selection",
                    options=[
                        {'label': 'Upload a CSV/TSV data set', 'value': 'user'},
                        {'label': 'ABCD release 4.0', 'value': 'nda.abcd'},
                        {'label': 'Early psychosis study', 'value': 'nda.earlypsychosis'},
                        {'label': 'HCP Ageing', 'value': 'nda.hcpageing'},
                    ],
                    value='user',
                ),
                style={"width": "30%"},
            ),
            user_upload.UserUpload(app),
            std_data.StdData(app, style={"display" : "none"}),
            html.Div(id="df-loaded-div", className="hidden"),
        ])

        self.register_cb(app, "data_loaded", 
            Output("df-loaded-div", "children"),
            Input("std-df-loaded-div", "children"),
            Input("user-df-loaded-div", "children"),
            prevent_initial_call=True,
        )
        
        self.register_cb(app, "data_selection_changed", 
            [
                Output("upload", "style"),
                Output("std", "style"),
            ],
            [
                Input("dataset-selection", "value")
            ],
            prevent_initial_call=True,
        )

    def data_loaded(self, std_loaded, user_loaded):
        return std_loaded or user_loaded
        
    def data_selection_changed(self, selection):
        """
        Change to source of data selection, user or standard data
        """
        self.log.info(f"Data selection: {selection}")
        data_store.set_session_data_store(selection)
        if selection == "user":
            return {"display" : "block"}, {"display" : "none"}
        else:
            return {"display" : "none"}, {"display" : "block"}
