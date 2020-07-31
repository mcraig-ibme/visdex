import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH
import plotly.graph_objects as go
from psych_dashboard.app import app, graph_types, dd_scatter_dims, input_scatter_dims, dd_bar_dims, style_dict


def generate_scatter_group(n_clicks):
    print('generate_scatter_group')
    return html.Div(id={'type': 'filter-graph-group-scatter',
                        'index': n_clicks
                        },
                    children=[html.Div([value + ":", dcc.Dropdown(id={'type': 'scatter_'+str(key), 'index': n_clicks},
                                                                  options=[])],
                                       id={'type': 'div_scatter_'+str(key), 'index': n_clicks},
                                       style=style_dict
                                       )
                              for key, value in dd_scatter_dims.items()
                              ]
                    + [html.Div([value + ":", dcc.Input(id={'type': 'scatter_'+str(key), 'index': n_clicks},
                                                        type='number',
                                                        min=0,
                                                        step=1,
                                                        )
                                 ],
                                id={'type': 'div_scatter_'+str(key), 'index': n_clicks},
                                style=style_dict)
                       for key, value in input_scatter_dims.items()
                       ]
                    + [dcc.Graph(id={'type': 'gen_scatter_graph', 'index': n_clicks},
                                 figure=go.Figure(data=go.Scatter()))
                       ]
                    )


def generate_bar_group(n_clicks):
    print('generate_bar_group')
    return html.Div(id={'type': 'filter-graph-group-bar',
                        'index': n_clicks
                        },
                    children=[html.Div([value + ":", dcc.Dropdown(id={'type': 'bar_'+key, 'index': n_clicks},
                                                                  options=[])],
                                       id={'type': 'div_bar_'+str(key), 'index': n_clicks},
                                       style=style_dict
                                       )
                              for key, value in dd_bar_dims.items()
                              ]
                    + [dcc.Graph(id={'type': 'gen_bar_graph', 'index': n_clicks},
                                 figure=go.Figure(data=go.Bar()))
                       ]
                    )


@app.callback(
    Output({'type': 'divgraph-type-dd', 'index': MATCH}, 'children'),
    [Input({'type': 'graph-type-dd', 'index': MATCH}, 'value')],
    [State({'type': 'graph-type-dd', 'index': MATCH}, 'id'),
     State({'type': 'divgraph-type-dd', 'index': MATCH}, 'children')]
)
def change_graph_group_type(graph_type, id, children):
    print('change_graph_group_type', graph_type, id, children)
    # Check whether the value of the dropdown matches the type of the existing group. If it doesn't match, then
    # generate a new group of the right type.
    if graph_type == 'Bar' and children[-1]['props']['id']['type'] != 'filter-graph-group-bar':
        children[-1] = generate_bar_group(id['index'])
    elif graph_type == 'Scatter' and children[-1]['props']['id']['type'] != 'filter-graph-group-scatter':
        children[-1] = generate_scatter_group(id['index'])
    return children


@app.callback(
    Output('graph-group-container', 'children'),
    [Input('add-graph-button', 'n_clicks')],
    [State('graph-group-container', 'children')])
def add_graph_group(n_clicks, children):
    # Add a new graph group each time the button is clicked. The if None guard stops there being an initial graph.
    print('add_graph_group')
    if n_clicks is not None:
        # This dropdown controls what type of graph-group to display next to it.
        new_graph_type_dd = html.Div(['Graph type:',
                                      dcc.Dropdown(id={'type': 'graph-type-dd',
                                                       'index': n_clicks
                                                       },
                                                   options=[{'label': value,
                                                             'value': value
                                                             }
                                                            for value in graph_types
                                                            ],
                                                   value='Scatter',
                                                   style={'width': '50%'}
                                                   ),
                                      # This is a placeholder for the 'filter-graph-group-scatter' or
                                      # 'filter-graph-group-bar' to be placed here.
                                      # Because graph-type-dd above is set to Scatter, this will initially be
                                      # automatically filled with a filter-graph-group-scatter.
                                      # But on the initial generation of this object, we give it type 'placeholder' to
                                      # make it easy to check its value in change_graph_group_type()
                                      html.Div(id={'type': 'placeholder', 'index': n_clicks})
                                      ],
                                     id={'type': 'divgraph-type-dd', 'index': n_clicks},
                                     )

        children.append(new_graph_type_dd)

    return children
