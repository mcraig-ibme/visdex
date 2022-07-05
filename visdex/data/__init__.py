"""
visdex: Data selection

These components handle the loading of the main data
"""
from dash import html, dcc
from dash.dependencies import Input, Output

from visdex.common import Component
from visdex.data_stores import DATA_STORES
from . import user_upload, std_data

class DataSelection(Component):
    def __init__(self, app):
        Component.__init__(self, app, id_prefix="data-", children=[
            html.H2(children="Data selection"),
            html.Div(
                dcc.Dropdown(
                    id="dataset-selection",
                    options=[],
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

        self.register_cb(app, "update_stores", 
            Output("dataset-selection", "options"),
            Input('url', 'pathname'),
            prevent_initial_call=False,
        )

    def data_loaded(self, std_loaded, user_loaded):
        return std_loaded or user_loaded

    def update_stores(self, url_login):
        data_store_selections = []
        for id, ds_conf in DATA_STORES.items():
            self.log.info("%s: %s", id, ds_conf)
            if ds_conf["impl"].check_user():
                data_store_selections.append({"label" : ds_conf["label"], "value" : id})
        return data_store_selections
