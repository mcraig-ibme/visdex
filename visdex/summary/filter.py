"""
visdex: Data filtering component
"""
from dash import html, dcc, dash_table, callback_context
from dash.dependencies import Input, Output, State, ALL

from visdex.common import vstack, hstack, standard_margin_left, Collapsible
from visdex.data import data_store
from visdex.summary import data_preview
from visdex.common.timing import timing

class DataFilter(Collapsible):
    """
    Component that filters the rows/columns of the data
    """
    def __init__(self, app, id_prefix="filter-", title="Data Filtering"):
        """
        :param app: Dash application
        """
        Collapsible.__init__(self, app, id_prefix, title, children=[
            html.H3(children="Column filter", style=vstack),
            html.Div(
                id=id_prefix+"missing-values-filter",
                children=[
                    html.Label("Filter out all columns missing at least X percentage of rows:", style=hstack),
                    dcc.Input(
                        id=id_prefix+"missing-values-input",
                        type="number",
                        min=0,
                        max=100,
                        debounce=True,
                        value=None,
                        style=hstack,
                    ),
                ],
            ),        
            html.H3(children="Row filter", style=vstack),
            html.Div(
                id=id_prefix+"predicate-filters",
                children=[           
                    html.Div(id=id_prefix+"predicate-filter-container", children=[]),
                    html.Button(
                        "Add row condition",
                        id=id_prefix+"add-row-predicate-button",
                        style={
                            "margin-top": "10px",
                            "margin-left": standard_margin_left,
                            "margin-bottom": "40px",
                        },
                    ),
                    html.Div(id=id_prefix+"random-sample", children=[
                        html.Label("Return random sample of: ", style={"display" : "inline-block", "verticalAlign" : "middle"}),
                        dcc.Input(id=id_prefix+"random-sample-input", style={"width" : "300px", "display" : "inline-block", "verticalAlign" : "middle"}),
                    ]),
                ],
            ),
            data_preview.DataPreview(app, "Data preview (filtered)", id_prefix="preview-filtered", update_div_id="filtered-loaded-div", data_id=data_store.FILTERED),
            html.Div(id="filtered-loaded-div", style={"display": "none"}, children=[]),
        ])

        self.register_cb(app, "update_filtered_data",
            Output("filtered-loaded-div", "children"),
            [
                Input("df-loaded-div", "children"),
                Input(id_prefix+"missing-values-input", "value"),
                Input({"type": id_prefix+"predicate-filter-column", "index": ALL}, "value"),
                Input({"type": id_prefix+"predicate-filter-op", "index": ALL}, "value"),
                Input({"type": id_prefix+"predicate-filter-value", "index": ALL}, "value"),
                Input(id_prefix+"random-sample-input", "value"),
                Input(id_prefix+"predicate-filter-container", "children"),
            ],
            prevent_initial_call=True,
        )

        self.register_cb(app, "add_row_predicate",
            Output(id_prefix+"predicate-filter-container", "children"),
            [
                Input("df-loaded-div", "children"),
                Input(id_prefix+"add-row-predicate-button", "n_clicks"),
            ],
            State(id_prefix+"predicate-filter-container", "children"),
            prevent_initial_call=True,
        )

    @timing
    def update_filtered_data(self, df_loaded, missing_value_cutoff, pred_cols, pred_ops, pred_values, random_sample, _pred_children):
        self.log.info(f"update_filtered_data")
        ds = data_store.get()

        df = ds.load(data_store.MAIN_DATA)
        if df.empty:
            return False

        ds.missing_values_threshold = missing_value_cutoff
        predicates = list(zip(pred_cols, pred_ops, pred_values))
        try:
            random_sample_size = int(random_sample)
            self.log.info("Random sample size: %i", random_sample_size)
            predicates.append((None, "random", random_sample_size))
        except:
            pass
        ds.predicates = predicates

        return True

    def add_row_predicate(self, n_clicks, df_loaded, children):
        which_input = callback_context.triggered[0]['prop_id'].split('.')[0]
        if which_input == "df-loaded-div":
            self.log.info(f"reset row predicates")
            return []
        else:
            self.log.info(f"Add row predicate")
            if n_clicks:
                df = data_store.get().load(data_store.FILTERED)
                cols = list(df.columns) + list(df.index.names)
                self.log.info(cols, df.index.names)
                col_options = [
                    {'label': c, 'value': c}
                    for c in cols
                ]
                op_options = [
                    {'label': c, 'value': c}
                    for c in ["==", ">", ">=", "<", "<=", "!=", "contains", "Not empty"]
                ]
                new_predicate = html.Div(
                    [
                        html.Label("Column: ", style={"display" : "inline-block", "verticalAlign" : "middle"}),
                        dcc.Dropdown(
                            id={"type" : self.id_prefix+"predicate-filter-column", "index" : n_clicks},
                            options=col_options, value=None,
                            style={"width" : "300px", "display" : "inline-block", "verticalAlign" : "middle"},
                        ),
                        dcc.Dropdown(
                            id={"type" : self.id_prefix+"predicate-filter-op", "index" : n_clicks},
                            options=op_options, value="=",
                            style={"width" : "200px", "display" : "inline-block", "verticalAlign" : "middle"},
                        ),
                        dcc.Input(
                            id={"type" : self.id_prefix+"predicate-filter-value", "index" : n_clicks},
                            style={"width" : "300px", "display" : "inline-block", "verticalAlign" : "middle"},
                        ),
                    ],
                    id=self.id_prefix+"predicate-filter-%i" % n_clicks,
                )

                children.append(new_predicate)
        
            return children
