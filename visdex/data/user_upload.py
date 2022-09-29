"""
visdex: Component to allow user to upload a data set
"""
import datetime

from dash import html, dcc
from dash.dependencies import Input, Output, State

from visdex.common import Component
import visdex.data_stores
import visdex.session

class UserUpload(Component):
    def __init__(self, app, *args, **kwargs):
        self.ds = None
        Component.__init__(self, app, id_prefix="user-", children=[
            dcc.Upload(
                id="data-file-upload",
                children=html.Div([html.A(id="data-file", children="Drag and drop or click to select files")]),
                className="upload"
            ),
            html.H2(
                children="Column Filter",
            ),
            "Upload an optional file containing columns to select",
            dcc.Upload(
                id="filter-file-upload",
                children=html.Div([html.A(id="filter-file", children="Drag and drop or click to select files")]),
                className="upload"
            ),
            html.Button("Clear column filter", id="clear-filter-button", className="inline"), 
            html.Div(id="data-warning"),
            # Hidden div for holding the booleans identifying whether a DF is loaded
            html.Div(id="user-df-loaded-div", className="hidden"),
        ], id="upload", *args, **kwargs)

        self.register_cb(app, "datastore_selection_changed", 
            Output("upload", "style"),
            Input("datastore-selection", "value"),
            prevent_initial_call=True,
        )

        self.register_cb(app, "data_file_changed",
            [Output("data-file", "children"), Output("user-df-loaded-div", "children")],
            [Input("data-file-upload", "contents")],
            [
                State("data-file-upload", "filename"), 
                State("data-file-upload", "last_modified"),
                State("filter-file-upload", "contents"),
            ],
            prevent_initial_call=True,
        )

        self.register_cb(app, "filter_file_changed",
            [Output("filter-file", "children"), Output("data-warning", "children")],
            [Input("filter-file-upload", "contents")],
            [
                State("filter-file-upload", "filename"),
                State("filter-file-upload", "last_modified"),
                State("data-file-upload", "contents"),
                State("data-file-upload", "filename"),
            ],
            prevent_initial_call=True,
        )

        self.register_cb(app, "filter_clear_button_clicked",
            Output("filter-file-upload", "children"),
            Input("clear-filter-button", "n_clicks"),
            [
                State("data-file-upload", "contents"),
                State("data-file-upload", "filename"),
            ],
            prevent_initial_call=True,
        )

    def datastore_selection_changed(self, selection):
        """
        If standard data has been selected show the data set / field lists and repopulate them
        """
        if selection == "user":
            if self.ds is None:
                self.ds = visdex.data_stores.DATA_STORES["user"]["impl"]
            return {"display" : "block"}
        else:
            return {"display" : "none"}

    def data_file_changed(self, data_contents, data_filename, data_date, filter_contents):
        """
        When using a user data source, the upload file has been changed
        """
        sess = visdex.session.get()
        try:
            df = self.ds.load_file(self, data_contents, data_filename, filter_contents)
            sess.store(visdex.session.MAIN_DATA, df)
            # FIXME loaded with warning
            return f"{data_filename} loaded, last modified {datetime.datetime.fromtimestamp(data_date).strftime('%Y-%m-%d %H:%M:%S')}", True
        except Exception as e:
            self.log.error(f"{e}")
            sess.store(visdex.session.MAIN_DATA, None)
            return "File not loaded: " + str(e), False

    def filter_file_changed(self, filter_contents, filter_filename, filter_date, data_contents, data_filename):
        """
        When using user data source, the column filter file has changed
        """
        if filter_contents is None:
            return "File not loaded", ""
        sess = visdex.session.get()
        try:
            df = self.ds.load_file(self, data_contents, data_filename, filter_contents)
            sess.store(visdex.session.MAIN_DATA, df)
            return f"{filter_filename} loaded, last modified {datetime.datetime.fromtimestamp(filter_date).strftime('%Y-%m-%d %H:%M:%S')}", ""
        except Exception as e:
            self.log.error(f"{e}")
            return "File not loaded: " + str(e), ""

    def filter_clear_button_clicked(self, n_clicks, data_contents, data_filename):
        """
        When using user data source, the column filter file has been cleared
        """
        sess = visdex.session.get()
        try:
            df = self.ds.load_file(self, data_contents, data_filename, None)
            sess.store(visdex.session.MAIN_DATA, df)
            return html.Div([html.A(id="filter-file", children="Drag and drop or click to select files")])
        except Exception as e:
            self.log.error(f"{e}")
