"""
visdex: download options

Options to download data or just matching IDs
"""
import pandas as pd

from dash import html, dcc, callback_context as ctx, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from visdex.common import Collapsible, vstack, hstack
from visdex.data import data_store

class Download(Collapsible):

    def __init__(self, app, id_prefix="download-"):
        """
        :param app: Dash application
        """
        Collapsible.__init__(self, app, id_prefix, title="Download data", children=[
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
            html.Button(
                    "Download imaging data links",
                    id=id_prefix + "links-button",
                    style=hstack,
                ),
            dbc.Modal(
            [
                dbc.ModalHeader("Imaging Data"),
                dbc.ModalBody(
                    [
                        dbc.Label("Imaging data available:"),
                        dash_table.DataTable(id=id_prefix + "links-modal-table", columns=[{"name": "Description", "id": "text"}], row_selectable='multi', style_cell={'textAlign': 'left'}),
                    ]
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button("OK", color="primary", id=id_prefix + "links-modal-ok"),
                        dbc.Button("Cancel", id=id_prefix + "links-modal-cancel"),
                    ]
                ),
            ],
            id=id_prefix + "links-modal",
            size="xl",
        ),
            dcc.Download(id=id_prefix + "download-data"),
            dcc.Download(id=id_prefix + "download-ids"),
            dcc.Download(id=id_prefix + "download-links"),
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

        self.register_cb(app, "show_links_modal",
            [
                Output(id_prefix + "links-modal", "is_open"),
                Output(id_prefix + "links-modal-table", "data"),
                Output(id_prefix + "download-links", "data"),
            ],
            [
                Input(id_prefix + "links-button", "n_clicks"),
                Input(id_prefix + "links-modal-ok", "n_clicks"),
                Input(id_prefix + "links-modal-cancel", "n_clicks"),
            ],
            [
                State(id_prefix + "links-modal", "is_open"),
                State(id_prefix + "links-modal-table", "derived_virtual_data"),
                State(id_prefix + "links-modal-table", "derived_virtual_selected_rows"),
            ],
            #Output(id_prefix + "download-links", "data"),
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

    def show_links_modal(self, show_n_clicks, ok_n_clicks, cancel_n_clicks, is_open, imaging_types, selected_rows):
        """
        Show modal for downloading imaging data links
        """
        triggered_ids = [c["prop_id"] for c in ctx.triggered]
        if triggered_ids[0] == self.id_prefix + "links-button.n_clicks":
            # Initial show of modal dialog
            ds = data_store.get()
            imaging_types = ds.imaging_types.to_dict('records')
            return True, imaging_types, None
        elif triggered_ids[0] == self.id_prefix + "links-modal-ok.n_clicks":
            # Ok clicked
            return False, [], self.download_links(data_store.get().imaging_types, selected_rows)
        else:
            # Cancel clicked
            return False, [], None

    def download_links(self, imaging_types, selected_rows):
        self.log.debug("Download imaging links")
        imaging_types = imaging_types.iloc[selected_rows]
        ids = pd.DataFrame(index=data_store.get().load(data_store.FILTERED).index)
        df = data_store.get().imaging_links(ids, imaging_types)
        return dcc.send_data_frame(df.to_csv, "visdex_imaging_links.csv")
