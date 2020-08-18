import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH
import plotly.graph_objects as go
from psych_dashboard.app import app, graph_types, all_scatter_components, all_bar_components, all_manhattan_components, style_dict


def generate_scatter_group(n_clicks):
    print('generate_scatter_group', n_clicks)
    children = list()
    for component in all_scatter_components:
        name = component['name']
        if component['component_type'] == 'Dropdown':
            children.append(html.Div([component['label'] + ":", dcc.Dropdown(id={'type': 'scatter_'+str(name), 'index': n_clicks},
                                                                  options=[])],
                                       id={'type': 'div_scatter_'+str(name), 'index': n_clicks},
                                       style=style_dict
                                       ))
        elif component['component_type'] == 'Input':
            children.append(html.Div([component['label'] + ":", dcc.Input(id={'type': 'scatter_'+str(name), 'index': n_clicks},
                                                                          type='number',
                                                                          min=0,
                                                                          step=1,)],
                                       id={'type': 'div_scatter_'+str(name), 'index': n_clicks},
                                       style=style_dict
                                       ))
    children.append(dcc.Graph(id={'type': 'gen_scatter_graph', 'index': n_clicks},
                    figure=go.Figure(data=go.Scatter()))
                    )
    print('children exploratory', children)

    return html.Div(id={'type': 'filter-graph-group-scatter',
                        'index': n_clicks
                        },
                    children=children
                    )


def generate_bar_group(n_clicks):
    print('generate_bar_group')
    children = list()
    for component in all_bar_components:
        if component['component_type'] == 'Dropdown':
            print(component, 'Dropdown')
            children.append(html.Div([component['label'] + ":",
                                      dcc.Dropdown(id={'type': 'bar_' + str(component['name']), 'index': n_clicks},
                                                   options=[])],
                                     id={'type': 'div_bar_' + str(component['name']), 'index': n_clicks},
                                     style=style_dict
                                     ))
        elif component['component_type'] == 'Input':
            print(component, 'Input')
            children.append(html.Div([component['label'] + ":",
                                      dcc.Input(id={'type': 'bar_' + str(component['name']), 'index': n_clicks},
                                                type='number',
                                                min=0,
                                                step=1, )],
                                     id={'type': 'div_bar_' + str(component['name']), 'index': n_clicks},
                                     style=style_dict
                                     ))
    children.append(dcc.Graph(id={'type': 'gen_bar_graph', 'index': n_clicks},
                              figure=go.Figure(data=go.Bar()))
                    )
    print(children)

    return html.Div(id={'type': 'filter-graph-group-bar',
                        'index': n_clicks
                        },
                    children=children
                    )


def generate_manhattan_group(n_clicks):
    print('generate_manhattan_group')
    children = list()
    for component in all_manhattan_components:
        if component['component_type'] == 'Dropdown':
            print(component, 'Dropdown')
            children.append(html.Div([component['label'] + ":",
                                      dcc.Dropdown(id={'type': 'manhattan_' + str(component['name']), 'index': n_clicks},
                                                   options=[])],
                                     id={'type': 'div_manhattan_' + str(component['name']), 'index': n_clicks},
                                     style=style_dict
                                     ))
        elif component['component_type'] == 'Input':
            print(component, 'Input')
            children.append(html.Div([component['label'] + ":",
                                      dcc.Input(id={'type': 'manhattan_' + str(component['name']), 'index': n_clicks},
                                                type=component['other_args']['type'],
                                                min=component['other_args']['min'],
                                                step=component['other_args']['step'],
                                                )],
                                     id={'type': 'div_manhattan_' + str(component['name']), 'index': n_clicks},
                                     style=style_dict
                                     ))
        elif component['component_type'] == 'Checklist':
            print(component, 'Checklist')
            children.append(html.Div([component['label'] + ":",
                                      dcc.Checklist(id={'type': 'manhattan_' + str(component['name']), 'index': n_clicks},
                                                    options=component['other_args']['options'],
                                                    value=component['other_args']['value'],
                                                    )],
                                     id={'type': 'div_manhattan_' + str(component['name']), 'index': n_clicks},
                                     style=style_dict
                                     ))
    children.append(dcc.Graph(id={'type': 'gen_manhattan_graph', 'index': n_clicks},
                              figure=go.Figure(data=go.Scatter()))
                    )
    print(children)

    return html.Div(id={'type': 'filter-graph-group-manhattan',
                        'index': n_clicks
                        },
                    children=children
                    )
    # return html.Div(id={'type': 'filter-graph-group-manhattan',
    #                     'index': n_clicks
    #                     },
    #                 children=[html.Div([value + ":", dcc.Dropdown(id={'type': 'manhattan_'+key, 'index': n_clicks},
    #                                                               options=[])],
    #                                    id={'type': 'div_manhattan_'+str(key), 'index': n_clicks},
    #                                    style=style_dict
    #                                    )
    #                           for key, value in dd_manhattan_dims.items()
    #                           ]
    #                 + [html.Div([value + ":", dcc.Input(id={'type': 'manhattan_' + str(key), 'index': n_clicks},
    #                                                     type='number',
    #                                                     value=0.05,
    #                                                     min=0,
    #                                                     step=0.001
    #                                                     )
    #                              ],
    #                             id={'type': 'div_manhattan_' + str(key), 'index': n_clicks},
    #                             style=style_dict)
    #                     for key, value in input_manhattan_dims.items()
    #                    ]
    #                 + [html.Div([value + ":", dcc.Checklist(id={'type': 'manhattan_' + str(key), 'index': n_clicks},
    #                                                         options=[{'label': '', 'value': 'LOG'}],
    #                                                         value=['LOG'],
    #                                                         style={'display': 'inline-block', 'width': '10%'}
    #                                                         )
    #                              ],
    #                             id={'type': 'div_manhattan_' + str(key), 'index': n_clicks},
    #                             style=style_dict)
    #                     for key, value in check_manhattan_dims.items()
    #                    ]
    #                 + [dcc.Graph(id={'type': 'gen_manhattan_graph', 'index': n_clicks},
    #                              figure=go.Figure(data=go.Scatter()))
    #                    ]
    #                 )


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
    elif graph_type == 'Manhattan' and children[-1]['props']['id']['type'] != 'filter-graph-group-manhattan':
        children[-1] = generate_manhattan_group(id['index'])
    print('children change', children)
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
