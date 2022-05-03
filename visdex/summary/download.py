"""
visdex: download options

Options to download data or just matching IDs
"""
import pandas as pd

from dash import html, dcc
from dash.dependencies import Input, Output

from visdex.common import Component, vstack, hstack
from visdex.data import data_store

class Download(Component):

    def __init__(self, app, id_prefix="download-"):
        """
        :param app: Dash application
        """
        Component.__init__(self, app, id_prefix, children=[
            html.H3(children="Download data", style=vstack),
            html.Button(
                    "Download data",
                    id=id_prefix + "data-button",
                    style=hstack,
                ),
            html.Button(
                    "Download IDs",
                    id=id_prefix + "ids-button",
                    style=hstack,
                ),
            dcc.Download(id=id_prefix + "download-data"),
            dcc.Download(id=id_prefix + "download-ids"),
        ])

        self.register_cb(app, "download_data",
            Output(id_prefix + "download-data", "data"),
            Input(id_prefix + "data-button", "n_clicks"),
            prevent_initial_call=True,
        )

        self.register_cb(app, "download_ids",
            Output(id_prefix + "download-ids", "data"),
            Input(id_prefix + "ids-button", "n_clicks"),
            prevent_initial_call=True,
        )

    def download_data(self, n_clicks):
        self.log.debug("Download data")
        df = data_store.get().load(data_store.FILTERED)
        return dcc.send_data_frame(df.to_csv, "visdex_data.csv")

    def download_ids(self, n_clicks):
        self.log.debug("Download IDs")
        df = pd.DataFrame(index=data_store.get().load(data_store.FILTERED).index)
        return dcc.send_data_frame(df.to_csv, "visdex_ids.csv")
