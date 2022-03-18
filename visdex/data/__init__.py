"""
visdex: Data handling components

These components handle the loading of the main data and filter files
"""
import logging
import io
import base64
import datetime

import pandas as pd

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from visdex.common import vstack, hstack, standard_margin_left
from .cache import get_cache, init_cache

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
            ],
            style={"display" : "block"},
        ),
        html.Div(
            id="std-visible",
            children=[
                html.Div(
                    id="visit-filter-heading",
                    children=[
                        dbc.Button(
                            "+",
                            id="collapse-visit-filter",
                            style={
                                "display": "inline-block",
                                "margin-left": "10px",
                                "width": "40px",
                                "vertical-align" : "middle",
                            },
                        ),
                        html.H3(
                            "Filter by visit",
                            style={
                                "display": "inline-block",
                                "margin-left": standard_margin_left,
                                "vertical-align" : "middle",
                                "margin-bottom" : "0",
                                "padding" : "0",
                            },
                        ),
                    ],
                ),
                 dbc.Collapse(
                    id="visit-filter",
                    children=[
                        html.Label("Selected visits: "),
                    ],
                    is_open=False,
                ),
                html.Div(
                    id="std-datasets",
                    children=[
                        html.Label("ABCD data sets"),
                        dash_table.DataTable(id="std-dataset-checklist", columns=[{"name": "Name", "id": "name"}], row_selectable='multi'),
                    ],
                    style={
                        "width": "70%", "height" : "300px", "overflow-y" : "scroll",
                        "display" : "inline-block",
                    }
                ),
                html.Div(
                    id="std-fields",
                    children=[
                        html.Label("Data set fields"),
                        dash_table.DataTable(id="std-field-checklist", columns=[{"name": "Field", "id": "ElementName"}], row_selectable='multi'),
                    ],
                    style={
                        "width": "20%", "height" : "300px", "overflow-y" : "scroll",
                        "display" : "inline-block",
                    }
                ),
            ],
            style={"display" : "none"},
        ),
        html.Button("Analyse", id="load-files-button", style=hstack),        
        html.Div(
            id="data-warning", children=[""], style={"color" : "blue", "display" : "inline-block", "margin-left" : "10px"}
        ),

        # Hidden divs for holding the booleans identifying whether a DF is loaded in
        # each case
        html.Div(id="df-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="df-filtered-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="corr-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="pval-loaded-div", style={"display": "none"}, children=[]),
    ])

    @app.callback(
        [Output("upload-visible", "style"), Output("std-visible", "style"), Output("std-dataset-checklist", "data")],
        [Input("dataset-selection", "value")],
        prevent_initial_call=True,
    )
    def data_selection_changed(selection):
        LOG.info(f"Data selection: {selection}")
        if selection == "user":
            init_cache()
            return {"display" : "block"}, {"display" : "none"}, []
        else:
            init_cache(selection)
            dataset_df = get_cache().get_datasets()
            #datasets = [{
            #    "label" : "%s: %s" % (r["name"], r["desc"]),
            #    "value" : r["short_name"]
            #} for idx, r in dataset_df.iterrows()]
            #print(datasets)
            return {"display" : "none"}, {"display" : "block"}, dataset_df.to_dict('records')

    @app.callback(
        Output("std-field-checklist", "data"),
        Input("std-dataset-checklist", "derived_virtual_data"),
        Input("std-dataset-checklist", "derived_virtual_selected_rows")
    )
    def dataset_selection_changed(data, selected_rows):
        selected_datasets = [data[idx]["short_name"] for idx in selected_rows]
        fields = get_cache().get_fields_in_datasets(selected_datasets)
        print(fields)
        #print(fields.to_dict('records'))
        return fields.to_dict('records')

    @app.callback(
        [Output("data-file", "children")],
        [Input("data-file-upload", "contents")],
        [State("data-file-upload", "filename"), State("data-file-upload", "last_modified")],
        prevent_initial_call=True,
    )
    def data_file_changed(contents, filename, date):
        """
        Handle data file upload
        
        Parses the contents of the uploaded file into a DataFrame and stores
        it in the cache, updating the appropriate children
        """
        LOG.info(f"parse data")
        cache = get_cache()

        if contents is not None:
            _content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)

            try:
                if filename.endswith("csv"):
                    # Assume that the user uploaded a CSV file
                    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
                elif filename.endswith("txt"):
                    # Assume that the user uploaded a whitespace-delimited CSV file
                    df = pd.read_csv(
                        io.StringIO(decoded.decode("utf-8")), delim_whitespace=True
                    )
                elif filename.endswith("xlsx"):
                    # Assume that the user uploaded an excel file
                    df = pd.read_excel(io.BytesIO(decoded))
                else:
                    raise NotImplementedError
            except Exception as e:
                LOG.error(f"{e}")
                return ["There was an error processing this file"]

            cache.set_main(df)

            return [
                f"{filename} loaded, last modified "
                f"{datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')}"
            ]

        return ["No file loaded", ""]

    @app.callback(
        [Output("filter-file", "children")],
        [Input("filter-file-upload", "contents")],
        [
            State("filter-file-upload", "filename"),
            State("filter-file-upload", "last_modified"),
        ],
        prevent_initial_call=True,
    )
    def _filter_file_changed(contents, filename, date):
        """
        Handle upload of column filter file

        The file is parsed and stored in the cache. It is used to filter
        the columns of the main data file, and the filtered data frame is
        also stored in the cache
        """
        LOG.info(f"parse filter")
        cache = get_cache()

        if contents is not None:
            _content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string).decode()

            try:
                variables_of_interest = [str(item) for item in decoded.splitlines()]
            except Exception as e:
                LOG.error(f"{e}")
                return ["There was an error processing this file"]

            cache.set_columns(variables_of_interest)

            return [
                f"{filename} loaded, last modified "
                f"{datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')}"
            ]

        return ["No file loaded"]

    @app.callback(
        Output("filter-file-upload", "children"),
        [Input("clear-filter-button", "n_clicks")],
        prevent_initial_call=True,
    )
    def clear_filter_button_clicked(n_clicks):
        LOG.info(f"clear filter")
        cache = get_cache()
        cache.set_columns()
        return html.Div([html.A(id="filter-file", children="Drag and drop or click to select files")]),

    @app.callback(
        [Output("df-loaded-div", "children"), Output("data-warning", "children")],
        [Input("load-files-button", "n_clicks")],
        [State("data-file-upload", "filename"), State("filter-file-upload", "filename")],
        prevent_initial_call=True,
    )
    def _analyse_button_clicked(n_clicks, data_file_value, filter_file_value):
        """
        Handle click on the 'Analyse' button
        
        The main data file is parsed and filtered and the result stored in the cache
        """
        LOG.info(f"update_df_loaded_div")
        cache = get_cache()
        warning = cache.filter()
        if not warning:
            warning = "Initial analysis complete"

        return [True, warning]

    @app.callback(
        [
            Output("visit-filter", "is_open"),
            Output("collapse-visit-filter", "children"),
        ],
        [Input("collapse-visit-filter", "n_clicks")],
        [State("visit-filter", "is_open")],
        prevent_initial_call=True,
    )
    def _toggle_visit_filter(n_clicks, is_open):
        """
        Handle click on the 'visit filter' expand/collapse button
        """
        LOG.info(f"toggle_visit filter {n_clicks} {is_open}")
        if n_clicks:
            return not is_open, "+" if is_open else "-"
        return is_open, "-"

    return layout
