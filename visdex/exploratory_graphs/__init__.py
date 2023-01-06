"""
visdex: Exploratory graphs

The exploratory graphs section defines specialised data visualisations that
can be generated by the user on request
"""
from dash import html, dcc
from dash.dependencies import Input, Output, State, MATCH
import plotly.graph_objects as go

from visdex.common import Collapsible

from .common import all_components, add_graph_type
from . import bar, histogram, manhattan, scatter, violin

class ExploratoryGraphs(Collapsible):
    def __init__(self, app, id_prefix="exp-"):
        """
        :param app: Dash application
        """
        Collapsible.__init__(self, app, id_prefix, title="Exploratory Graphs", children=[
            # Container to hold all the exploratory graphs
            html.Div(id="graph-group-container", children=[]),
            # Button at the page bottom to add a new graph
            html.Button(
                "New Graph",
                id="add-graph-button",
            ),
        ], is_open=True)

        self.register_cb(app, "add_graph", 
            Output("graph-group-container", "children"),
            [Input("add-graph-button", "n_clicks")],
            [State("graph-group-container", "children")],
            prevent_initial_call=True,
        )

        self.register_cb(app, "change_graph_type", 
            Output({"type": "divgraph-type-dd", "index": MATCH}, "children"),
            [Input({"type": "graph-type-dd", "index": MATCH}, "value")],
            [
                State({"type": "graph-type-dd", "index": MATCH}, "id"),
                State({"type": "divgraph-type-dd", "index": MATCH}, "children"),
            ],
        )
            
        add_graph_type(app, "bar", bar.make_figure)
        add_graph_type(app, "histogram", histogram.make_figure)
        add_graph_type(app, "manhattan", manhattan.make_figure)
        add_graph_type(app, "scatter", scatter.make_figure)
        add_graph_type(app, "violin", violin.make_figure)
        
    def add_graph(self, n_clicks, children):
        # Add a new graph group each time the button is clicked. The if None guard stops
        # there being an initial graph.
        if n_clicks is not None:
            # This dropdown controls what type of graph-group to display next to it.
            self.log.debug(f"Creating graph {n_clicks}")
            new_graph = html.Div(
                [
                    "Graph type:",
                    dcc.Dropdown(
                        id={"type": "graph-type-dd", "index": n_clicks},
                        options=[
                            {"label": str(value).capitalize(), "value": value}
                            for value in all_components.keys()
                        ],
                        value="scatter",
                    ),
                    # This is a placeholder for the 'filter-graph-group-scatter' or
                    # 'filter-graph-group-bar' to be placed here.
                    # Because graph-type-dd above is set to Scatter, this will initially be
                    # automatically filled with a filter-graph-group-scatter.
                    # But on the initial generation of this object, we give it type
                    # 'placeholder' to make it easy to check its value in
                    # change_graph_group_type()
                    html.Div(id={"type": "placeholder", "index": n_clicks}),
                ],
                id={"type": "divgraph-type-dd", "index": n_clicks},
            )

            children.append(new_graph)

        return children

    def change_graph_type(self, graph_type, id, children):
        self.log.info(f"change_graph_type {graph_type} {id}")
        if "filter-graph-group-" + str(graph_type) != children[-1]["props"]["id"]["type"]:
            graph_idx = id["index"]
            graph_children = []
            component_list = all_components[graph_type]
            for component in component_list:
                name = component["id"]
                args_to_replicate = dict(component)
                del args_to_replicate["component_type"]
                del args_to_replicate["id"]
                del args_to_replicate["label"]

                # Generate each component with the correct id, index, and arguments, inside its
                # own Div.
                graph_children.append(
                    html.Div(
                        [
                            component["label"] + ":",
                            component["component_type"](
                                id={"type": graph_type + "-" + name, "index": graph_idx},
                                **args_to_replicate,
                            ),
                        ],
                        id={"type": "div-" + graph_type + "-" + name, "index": graph_idx},
                        className="plot"
                    )
                )

            graph_children.append(
                dcc.Graph(
                    id={"type": "gen-" + graph_type + "-graph", "index": graph_idx},
                    figure=go.Figure(data=go.Scatter()),
                )
            )

            children[-1] = html.Div(
                id={"type": "filter-graph-group-" + graph_type, "index": graph_idx},
                children=graph_children,
            )

        return children
