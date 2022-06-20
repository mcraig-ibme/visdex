"""
visdex: Component to browse standard data
"""
from dash import html, dash_table
from dash.dependencies import Input, Output, State

from visdex.common import Component
from . import data_store

class StdData(Component):
    def __init__(self, app, id_prefix="std-", *args, **kwargs):
        Component.__init__(self, app, id_prefix, children=[
            html.Div(
                id=id_prefix+"dataset-select",
                children=[
                    html.H3("Data sets"),
                    html.Div(
                        id=id_prefix+"datasets",
                        children=[
                            dash_table.DataTable(id=id_prefix+"dataset-checklist", columns=[{"name": "Name", "id": "title"}], row_selectable='multi', style_cell={'textAlign': 'left'}),
                        ]
                    ),
                    html.Div(id=id_prefix+"dataset-desc"),
                ]
            ),
            
            html.Div(
                id=id_prefix+"field-select",
                children=[
                    html.H3("Data set fields"), 
                    html.Div(
                        id=id_prefix+"fields",
                        children=[
                            dash_table.DataTable(id=id_prefix+"field-checklist", columns=[{"name": "Field", "id": "ElementName"}], row_selectable='multi', style_cell={'textAlign': 'left'}),
                        ]
                    ),
                    html.Div(id=id_prefix+"field-desc"),
                ]
            ),
            html.Button("Load Data", id=id_prefix+"load-button"),
            html.Div(id=id_prefix+"df-loaded-div", className="hidden"),
        ], id="std", *args, **kwargs),

        self.register_cb(app, "std_data_selected", 
            Output(id_prefix+"dataset-checklist", "data"),
            Input("std", "style"),
            prevent_initial_call=True,
        )

        self.register_cb(app, "dataset_selection_changed",
            Output(id_prefix+"field-checklist", "data"),
            Output(id_prefix+"field-checklist", "selected_rows"),
            Input(id_prefix+"dataset-checklist", "derived_virtual_data"),
            Input(id_prefix+"dataset-checklist", "derived_virtual_selected_rows"),
            State(id_prefix+"field-checklist", "derived_virtual_data"),
            State(id_prefix+"field-checklist", "derived_virtual_selected_rows"),
            State("dataset-selection", "value"),
            prevent_initial_call=True,
        )
        
        self.register_cb(app, "dataset_active_changed",
            Output(id_prefix+"dataset-desc", "children"),
            Input(id_prefix+"dataset-checklist", "derived_virtual_data"),
            Input(id_prefix+"dataset-checklist", "active_cell"),
            prevent_initial_call=True,
        )
        
        self.register_cb(app, "field_active_changed",
            Output(id_prefix+"field-desc", "children"),
            Input(id_prefix+"field-checklist", "derived_virtual_data"),
            Input(id_prefix+"field-checklist", "active_cell"),
            prevent_initial_call=True,
        )

        self.register_cb(app, "load_button_clicked",
            Output(id_prefix+"df-loaded-div", "children"),
            Input(id_prefix+"load-button", "n_clicks"),
            State(id_prefix+"field-checklist", "derived_virtual_data"),
            State(id_prefix+"field-checklist", "derived_virtual_selected_rows"),
            prevent_initial_call=True,
        )

    def std_data_selected(self, style):
        """
        Standard data may have been selected by the user - need to repopulate dataset list
        """
        if style["display"] == "block":
            dataset_df = data_store.get().get_all_datasets()
            return dataset_df.to_dict('records')
        else:
            return []

    def dataset_selection_changed(self, data, selected_rows, field_data, selected_field_rows, data_type):
        """
        When using a standard data source, the set of selected data sets have been changed
        """
        # Not sure why this is needed when we have prevent_initial_call=True?
        if data_type == "user":
            self.log.error('dataset_selection_changed fired although we are in user data mode')
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
            self.log.exception('Error changing dataset')
            return [], []

    def dataset_active_changed(self, data, active_cell):
        """
        When using a standard data source, the active (clicked on) selected data set has been changed
        """
        try:
            return data[active_cell["row"]]["desc"]
        except (TypeError, KeyError, IndexError):
            return ""

    def field_active_changed(self, data, active_cell):
        """
        When using a standard data source, the active (clicked on) selected field has been changed
        """
        try:
            return data[active_cell["row"]]["ElementDescription"]
        except (TypeError, IndexError):
            return ""

    def load_button_clicked(self, n_clicks, fields, selected_rows):
        """
        When using standard data, the load button is clicked
        """
        self.log.debug(fields)
        self.log.debug(selected_rows)
        selected_fields = [fields[idx]["ElementName"] for idx in selected_rows]
        try:
            data_store.get().fields = selected_fields
            return True
        except Exception as e:
            self.log.exception('Error loading dataset')
            return False
