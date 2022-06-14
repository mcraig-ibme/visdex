"""
visdex: Component to browse standard data

These components handle the loading of the main data and filter files
"""
from dash import html, dash_table
from dash.dependencies import Input, Output, State

from visdex.common import Component
from . import data_store

class StdData(Component):
    def __init__(self, app, *args, **kwargs):
        Component.__init__(self, app, id_prefix="data-", children=[
            html.Div(
                id="std-dataset-select",
                children=[
                    html.H3("Data sets"),
                    html.Div(
                        id="std-datasets",
                        children=[
                            dash_table.DataTable(id="std-dataset-checklist", columns=[{"name": "Name", "id": "title"}], row_selectable='multi', style_cell={'textAlign': 'left'}),
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
            html.Div(id="std-df-loaded-div", style={"display": "none"}, children=[]),
        ], id="std", *args, **kwargs),

        self.register_cb(app, "std_dataset_selection_changed",
            Output("std-field-checklist", "data"),
            Output("std-field-checklist", "selected_rows"),
            Input("std-dataset-checklist", "derived_virtual_data"),
            Input("std-dataset-checklist", "derived_virtual_selected_rows"),
            State("std-field-checklist", "derived_virtual_data"),
            State("std-field-checklist", "derived_virtual_selected_rows"),
            State("dataset-selection", "value"),
            prevent_initial_call=True,
        )
        
        self.register_cb(app, "std_dataset_active_changed",
            Output("std-dataset-desc", "children"),
            Input("std-dataset-checklist", "derived_virtual_data"),
            Input("std-dataset-checklist", "active_cell"),
            prevent_initial_call=True,
        )
        
        self.register_cb(app, "std_field_active_changed",
            Output("std-field-desc", "children"),
            Input("std-field-checklist", "derived_virtual_data"),
            Input("std-field-checklist", "active_cell"),
            prevent_initial_call=True,
        )

        self.register_cb(app, "load_std_button_clicked",
            Output("std-df-loaded-div", "children"),
            Input("std-load-button", "n_clicks"),
            State("std-field-checklist", "derived_virtual_data"),
            State("std-field-checklist", "derived_virtual_selected_rows"),
            prevent_initial_call=True,
        )

    def std_dataset_selection_changed(self, data, selected_rows, field_data, selected_field_rows, data_type):
        """
        When using a standard data source, the set of selected data sets have been changed
        """
        self.log.info("Standard dataset changed")
        # Not sure why this is needed when we have prevent_initial_call=True?
        if data_type == "user":
            self.log.error('std_dataset_selection_changed fired although we are in user data mode')
            return [], []

        try:
            selected_datasets = [data[idx]["shortname"] for idx in selected_rows]
            data_store.get().datasets = selected_datasets
            fields = data_store.get().get_all_fields().to_dict('records')

            # Change the set of selected field rows so they match the same fields before the change
            selected_fields_cur = [field_data[idx]["ElementName"] for idx in selected_field_rows]
            selected_fields_new = [idx for idx, record in enumerate(fields) if record["ElementName"] in selected_fields_cur]

            self.log.debug("Fields found:")
            self.log.debug(fields)
            return fields, selected_fields_new
        except Exception as e:
            self.log.exception('Error changing std dataset')
            return [], []

    def std_dataset_active_changed(self, data, active_cell):
        """
        When using a standard data source, the active (clicked on) selected data set has been changed
        """
        self.log.info("Standard dataset active changed")
        try:
            return data[active_cell["row"]]["desc"]
        except (TypeError, KeyError, IndexError):
            return ""

    def std_field_active_changed(self, data, active_cell):
        """
        When using a standard data source, the active (clicked on) selected field has been changed
        """
        self.log.info("Standard dataset active field changed")
        try:
            return data[active_cell["row"]]["ElementDescription"]
        except (TypeError, IndexError):
            return ""

    def load_std_button_clicked(self, n_clicks, fields, selected_rows):
        """
        When using standard data, the load button is clicked
        """
        self.log.info(f"Load standard data {n_clicks}")
        self.log.debug(fields)
        self.log.debug(selected_rows)
        selected_fields = [fields[idx]["ElementName"] for idx in selected_rows]
        try:
            data_store.get().fields = selected_fields
            return True
        except Exception as e:
            self.log.exception('Error loading std dataset')
            return False
