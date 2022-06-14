"""
visdex: Data handling components

These components handle the loading of the main data and filter files
"""
import base64
import datetime

from dash import html, dcc
from dash.dependencies import Input, Output, State

from visdex.common import vstack, hstack, Component
from visdex.data import data_store

class UserUpload(Component):
    def __init__(self, app):
        Component.__init__(self, app, id_prefix="user-", children=[
            dcc.Upload(
                id="data-file-upload",
                children=html.Div([html.A(id="data-file", children="Drag and drop or click to select files")]),
                style={
                    "height": "60px",
                    "lineHeight" : "60px",
                    "borderWidth" : "1px",
                    "borderStyle" : "dashed",
                    "borderRadius" : "5px",
                    "textAlign" : "center",
                    "margin" : "10px",
                },
            ),
            html.H2(
                children="Column Filter",
                style=vstack,
            ),
            html.Label("Upload an optional file containing columns to select", style=hstack),
            html.Button("Clear", id="clear-filter-button", style=hstack), 
            dcc.Upload(
                id="filter-file-upload",
                children=html.Div([html.A(id="filter-file", children="Drag and drop or click to select files")]),
                style={
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "10px",
                },
            ),
            html.Div(
                id="data-warning", children=[""], style={"color" : "blue", "display" : "inline-block", "margin-left" : "10px"}
            ),
            # Hidden div for holding the booleans identifying whether a DF is loaded
            html.Div(id="user-df-loaded-div", style={"display": "none"}, children=[]),
        ], style={"display" : "block"}, id="upload")

        self.register_cb(app, "user_data_file_changed",
            [Output("data-file", "children"), Output("user-df-loaded-div", "children")],
            [Input("data-file-upload", "contents")],
            [State("data-file-upload", "filename"), State("data-file-upload", "last_modified")],
            prevent_initial_call=True,
        )

        self.register_cb(app, "user_filter_file_changed",
            [Output("filter-file", "children"), Output("data-warning", "children")],
            [Input("filter-file-upload", "contents")],
            [
                State("filter-file-upload", "filename"),
                State("filter-file-upload", "last_modified"),
            ],
            prevent_initial_call=True,
        )

        self.register_cb(app, "user_filter_clear_button_clicked",
            [Output("filter-file-upload", "children")],
            [Input("clear-filter-button", "n_clicks")],
            prevent_initial_call=True,
        )

    def user_data_file_changed(self, contents, filename, date):
        """
        When using a user data source, the upload file has been changed
        """
        self.log.info(f"data_file_changed")
        ds = data_store.get()
        try:
            ds.datasets = [(contents, filename, date)]
            # FIXME loaded with warning
            return f"{filename} loaded, last modified {datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')}"
        except Exception as e:
            self.log.error(f"{e}")
            return "File not loaded: " + str(e), False

    def user_filter_file_changed(self, contents, filename, date):
        """
        When using user data source, the column filter file has changed
        """
        self.log.info(f"filter_file_changed")
        ds = data_store.get()

        if contents is None:
            return "File not loaded", ""
        
        try:
            _content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string).decode()
            variables_of_interest = [str(item) for item in decoded.splitlines()]
            ds.fields = variables_of_interest
            return (
                f"{filename} loaded, last modified {datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')}",
                ""
            )
        except Exception as e:
            self.log.error(f"{e}")
            return "File not loaded: " + str(e), ""

    def user_filter_clear_button_clicked(self, n_clicks):
        """
        When using user data source, the column filter file has been cleared
        """
        self.log.info(f"clear filter")
        data_store.get().fields = []
        return html.Div([html.A(id="filter-file", children="Drag and drop or click to select files")])
