"""
visdex: Data handling components

These components handle the loading of the main data and filter files
"""
import logging
import io
import base64
import csv
import datetime

import pandas as pd

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from visdex.common import vstack, hstack, standard_margin_left
from visdex.data import data_store

LOG = logging.getLogger(__name__)

def get_layout(app):
    layout = html.Div(children=[
        html.H2(children="Data selection", style=vstack),
        html.Div(
            dcc.Dropdown(
                id="dataset-selection",
                options=[
                    {'label': 'Upload a CSV/TSV data set', 'value': 'user'},
                    {'label': 'ABCD data', 'value': 'abcd'},
                ],
                value='user',
            ),
            style={"width": "30%", "margin" : "10px"},
        ),
        html.Div(
            id="upload-visible",
            children=[
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
            ],
            style={"display" : "block"},
        ),
        html.Div(
            id="std-visible",
            children=[
                #html.Div(
                #    id="visit-filter-heading",
                #    children=[
                #        dbc.Button(
                #            "+",
                #            id="collapse-visit-filter",
                #            style={
                #                "display": "inline-block",
                #                "margin-left": "10px",
                #                "width": "40px",
                #                "vertical-align" : "middle",
                #            },
                #        ),
                #        html.H3(
                #            "Filter by visit",
                #            style={
                #                "display": "inline-block",
                #                "margin-left": standard_margin_left,
                #                "vertical-align" : "middle",
                #                "margin-bottom" : "0",
                #                "padding" : "0",
                #            },
                #        ),
                #    ],
                #),
                #dbc.Collapse(
                #    id="visit-filter",
                #    children=[
                #        html.Label("Selected visits: "),
                #    ],
                #    is_open=False,
                #),
                html.Div(
                    id="std-dataset-select",
                    children=[
                        html.H3("ABCD data sets"),
                        html.Div(
                            id="std-datasets",
                            children=[
                                dash_table.DataTable(id="std-dataset-checklist", columns=[{"name": "Name", "id": "name"}], row_selectable='multi', style_cell={'textAlign': 'left'}),
                            ],
                            style={
                                "width": "100%", "height" : "300px", "overflow-y" : "scroll",
                            }
                        ),
                        html.Div(
                            id="std-dataset-desc", children=[""], style={"color" : "blue", "display" : "inline-block", "margin-left" : "10px"}
                        ),
                    ],
                    style={"width": "55%", "display" : "inline-block"},
                ),
                
                html.Div(
                    id="std-field-select",
                    children=[
                        html.H3("Data set fields"), 
                        html.Div(
                            id="std-fields",
                            children=[
                                dash_table.DataTable(id="std-field-checklist", columns=[{"name": "Field", "id": "ElementName"}], row_selectable='multi', style_cell={'textAlign': 'left'}),
                            ],
                            style={
                                "width": "100%", "height" : "300px", "overflow-y" : "scroll",
                            }
                        ),
                        html.Div(
                            id="std-field-desc", children=[""], style={"color" : "blue", "display" : "inline-block", "margin-left" : "10px"}
                        ),
                    ],
                    style={"width": "40%", "display" : "inline-block"},
                ),
                html.Button("Load Data", id="std-load-button"),
            ],
            style={"display" : "none"},
        ),

        # Hidden divs for holding the booleans identifying whether a DF is loaded in
        # each case
        html.Div(id="std-df-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="user-df-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="df-loaded-div", style={"display": "none"}, children=[]),
    ])

    @app.callback(
        Output("df-loaded-div", "children"),
        Input("std-df-loaded-div", "children"),
        Input("user-df-loaded-div", "children"),
        prevent_initial_call=True,
    )
    def df_loaded(std, user):
        return std or user
        
    @app.callback(
        [Output("upload-visible", "style"), Output("std-visible", "style"), Output("std-dataset-checklist", "data")],
        [Input("dataset-selection", "value")],
        prevent_initial_call=True,
    )
    def data_selection_changed(selection):
        """
        Change to source of data selection, user or standard data
        """
        LOG.info(f"Data selection: {selection}")
        if selection == "user":
            data_store.init()
            return {"display" : "block"}, {"display" : "none"}, []
        else:
            data_store.init(selection)
            dataset_df = data_store.get().get_all_datasets()
            return {"display" : "none"}, {"display" : "block"}, dataset_df.to_dict('records')

    @app.callback(
        [Output("data-file", "children"), Output("user-df-loaded-div", "children")],
        [Input("data-file-upload", "contents")],
        [State("data-file-upload", "filename"), State("data-file-upload", "last_modified")],
        prevent_initial_call=True,
    )
    def user_data_file_changed(contents, filename, date):
        """
        When using a user data source, the upload file has been changed
        """
        LOG.info(f"data_file_changed")
        ds = data_store.get()

        if contents is not None:
            _content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)

            try:
                if filename.endswith("csv") or filename.endswith("txt"):
                    # Assume that the user uploaded a delimited file
                    dialect = csv.Sniffer().sniff(decoded.decode("utf-8")[:1024])
                    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep=dialect.delimiter, quotechar=dialect.quotechar, low_memory=False)
                elif filename.endswith("xlsx"):
                    # Assume that the user uploaded an excel file
                    df = pd.read_excel(io.BytesIO(decoded))
                else:
                    raise NotImplementedError
            except Exception as e:
                LOG.error(f"{e}")
                return ["There was an error processing this file", False]

            ds.store(data_store.MAIN_DATA, df)
            return [
                f"{filename} loaded, last modified {datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')}",
                True
            ]

        return ["No file loaded", False]

    @app.callback(
        [Output("filter-file", "children"), Output("data-warning", "children")],
        [Input("filter-file-upload", "contents")],
        [
            State("filter-file-upload", "filename"),
            State("filter-file-upload", "last_modified"),
        ],
        prevent_initial_call=True,
    )
    def user_filter_file_changed(contents, filename, date):
        """
        When using user data source, the column filter file has changed
        """
        LOG.info(f"filter_file_changed")
        ds = data_store.get()

        if contents is not None:
            _content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string).decode()

            try:
                variables_of_interest = [str(item) for item in decoded.splitlines()]
            except Exception as e:
                LOG.error(f"{e}")
                return ["There was an error processing this file"]

            warning = ds.set_column_filter(variables_of_interest)

            return [
                f"{filename} loaded, last modified "
                f"{datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')}",
                warning
            ]

        return ["No file loaded", ""]

    @app.callback(
        [Output("filter-file-upload", "children")],
        [Input("clear-filter-button", "n_clicks")],
        prevent_initial_call=True,
    )
    def user_filter_clear_button_clicked(n_clicks):
        """
        When using user data source, the column filter file has been cleared
        """
        LOG.info(f"clear filter")
        data_store.get().set_column_filter()
        return html.Div([html.A(id="filter-file", children="Drag and drop or click to select files")])

    @app.callback(
        Output("std-field-checklist", "data"),
        Output("std-field-checklist", "selected_rows"),
        Input("std-dataset-checklist", "derived_virtual_data"),
        Input("std-dataset-checklist", "derived_virtual_selected_rows"),
        State("std-field-checklist", "derived_virtual_data"),
        State("std-field-checklist", "derived_virtual_selected_rows")
    )
    def std_dataset_selection_changed(data, selected_rows, field_data, selected_field_rows):
        """
        When using a standard data source, the set of selected data sets have been changed
        """
        selected_datasets = [data[idx]["short_name"] for idx in selected_rows]
        data_store.get().select_datasets(selected_datasets)
        fields = data_store.get().get_all_fields().to_dict('records')

        # Change the set of selected field rows so they match the same fields before the change
        selected_fields_cur = [field_data[idx]["ElementName"] for idx in selected_field_rows]
        selected_fields_new = [idx for idx, record in enumerate(fields) if record["ElementName"] in selected_fields_cur]

        return fields, selected_fields_new

    @app.callback(
        Output("std-dataset-desc", "children"),
        Input("std-dataset-checklist", "derived_virtual_data"),
        Input("std-dataset-checklist", "active_cell"),
    )
    def std_dataset_active_changed(data, active_cell):
        """
        When using a standard data source, the active (clicked on) selected data set has been changed
        """
        return data[active_cell["row"]]["desc"]

    @app.callback(
        Output("std-field-desc", "children"),
        Input("std-field-checklist", "derived_virtual_data"),
        Input("std-field-checklist", "active_cell"),
    )
    def std_dataset_active_changed(data, active_cell):
        """
        When using a standard data source, the active (clicked on) selected field has been changed
        """
        if active_cell is not None and active_cell["row"] is not None:
            print(data[active_cell["row"]])
            return data[active_cell["row"]]["ElementDescription"]

    @app.callback(
        [
            Output("visit-filter", "is_open"),
            Output("collapse-visit-filter", "children"),
        ],
        [Input("collapse-visit-filter", "n_clicks")],
        [State("visit-filter", "is_open")],
        prevent_initial_call=True,
    )
    def std_toggle_visit_filter(n_clicks, is_open):
        """
        Handle click on the 'visit filter' expand/collapse button for standard data sets
        """
        LOG.info(f"toggle_visit filter {n_clicks} {is_open}")
        if n_clicks:
            return not is_open, "+" if is_open else "-"
        return is_open, "-"

    @app.callback(
        Output("std-df-loaded-div", "children"),
        Input("std-load-button", "n_clicks"),
        State("std-field-checklist", "derived_virtual_data"),
        State("std-field-checklist", "derived_virtual_selected_rows"),
        prevent_initial_call=True,
    )
    def load_std_button_clicked(n_clicks, fields, selected_rows):
        """
        When using standard data, the load button is clicked
        """
        LOG.info(f"Load standard data")
        selected_fields = [fields[idx] for idx in selected_rows]
        data_store.get().select_fields(selected_fields)
        return True

    return layout
