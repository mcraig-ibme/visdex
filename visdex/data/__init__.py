"""
visdex: Data selection

These components handle the loading of the main data
"""
from dash import html, dcc
from dash.dependencies import Input, Output

from visdex.common import vstack, Component
from . import data_store, user_upload, std_data

class DataSelection(Component):
    def __init__(self, app):
        Component.__init__(self, app, id_prefix="data-", children=[
            html.H2(children="Data selection", style=vstack),
            html.Div(
                dcc.Dropdown(
                    id="dataset-selection",
                    options=[
                        {'label': 'Upload a CSV/TSV data set', 'value': 'user'},
                        {'label': 'ABCD release 4.0', 'value': 'abcd'},
                        {'label': 'Early psychosis study', 'value': 'earlypsychosis'},
                        {'label': 'HCP Ageing', 'value': 'hcpageing'},
                    ],
                    value='user',
                ),
                style={"width": "30%", "margin" : "10px"},
            ),
            user_upload.UserUpload(app),
            std_data.StdData(app, style={"display" : "none"}),
            html.Div(id="df-loaded-div", style={"display": "none"}, children=[]),
        ])

        self.register_cb(app, "data_loaded", 
            Output("df-loaded-div", "children"),
            Input("std-df-loaded-div", "children"),
            Input("user-df-loaded-div", "children"),
            prevent_initial_call=True,
        )
        
        self.register_cb(app, "data_selection_changed", 
            [Output("upload", "style"), Output("std", "style"), Output("std-dataset-checklist", "data")],
            [Input("dataset-selection", "value")],
            prevent_initial_call=True,
        )

    def data_loaded(self, std_loaded, user_loaded):
        return std_loaded or user_loaded
        
    def data_selection_changed(self, selection):
        """
        Change to source of data selection, user or standard data
        """
        self.log.info(f"Data selection: {selection}")
        if selection == "user":
            data_store.init()
            return {"display" : "block"}, {"display" : "none"}, []
        else:
            data_store.init(selection)
            dataset_df = data_store.get().get_all_datasets()
            return {"display" : "none"}, {"display" : "block"}, dataset_df.to_dict('records')
