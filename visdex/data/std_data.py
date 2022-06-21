"""
visdex: Component to browse standard data
"""
from dash import html, dash_table
from dash.dependencies import Input, Output, State

import visdex.session
import visdex.common
from visdex.data_stores import DATA_STORES

class StdData(visdex.common.Component):
    def __init__(self, app, id_prefix="std-", *args, **kwargs):
        self.ds = None

        visdex.common.Component.__init__(self, app, id_prefix, children=[
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

        self.register_cb(app, "datastore_selection_changed", 
            Output("std", "style"),
            Output(id_prefix+"dataset-checklist", "data"),
            Input("dataset-selection", "value"),
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
            Input(id_prefix+"dataset-checklist", "derived_virtual_data"),
            Input(id_prefix+"dataset-checklist", "derived_virtual_selected_rows"),
            State(id_prefix+"field-checklist", "derived_virtual_data"),
            State(id_prefix+"field-checklist", "derived_virtual_selected_rows"),
            prevent_initial_call=True,
        )

    def datastore_selection_changed(self, selection):
        """
        If standard data has been selected show the data set / field lists and repopulate them
        """
        if selection != "user":
            sess = visdex.session.get()
            self.ds = DATA_STORES[selection]["impl"]
            sess.set_prop("ds", selection)
            dataset_df = self.ds.datasets
            return {"display" : "block"}, dataset_df.to_dict('records')
        else:
            self.ds = None
            return {"display" : "none"}, []

    def dataset_selection_changed(self, data, selected_rows, field_data, selected_field_rows, data_type):
        """
        When using a standard data source, the set of selected data sets have been changed
        """
        # Not sure why this is needed when we have prevent_initial_call=True?
        if data_type == "user":
            self.log.error('dataset_selection_changed fired although we are in user data mode')
            return [], []
        elif self.ds is None:
            self.log.error('dataset_selection_changed fired although ds is still None')
            return [], []

        try:
            selected_datasets = [data[idx]["shortname"] for idx in selected_rows]
            fields = self.ds.get_fields(*selected_datasets).to_dict('records')

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

    def load_button_clicked(self, n_clicks, dataset_info, dataset_selected_rows, field_info, field_selected_rows):
        """
        When using standard data, the load button is clicked
        """
        if self.ds is None:
            return False

        self.log.debug(field_info)
        self.log.debug(field_selected_rows)
        selected_fields = [field_info[idx]["ElementName"] for idx in field_selected_rows]
        try:
            sess = visdex.session.get()
            # FIXME more than one selected data set
            datasets = [dataset_info[idx]["shortname"] for idx in dataset_selected_rows]
            if datasets:
                sess.store(visdex.session.MAIN_DATA, self.ds.get_data(datasets[0], selected_fields))
                return True
            return False
        except Exception as e:
            self.log.exception('Error loading dataset')
            return False
