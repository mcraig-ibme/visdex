import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH
import plotly.graph_objects as go
from collections import defaultdict
from psych_dashboard.app import app, graph_types, all_scatter_components, all_bar_components, all_manhattan_components, style_dict


def create_arguments_nested_dict(components_list, args):
    # Generate the list of argument names based on the input order, paired by component id and property name
    keys = [(component['id'], str(prop))
            for component in components_list for prop in component]
    # Convert inputs to a nested dict, with the outer key the component id, and the inner key the property name
    args_dict = defaultdict(dict)
    for key, value in zip(keys, args):
        args_dict[key[0]][key[1]] = value
    return args_dict


def update_graph_components(graph_type, component_list, dd_options, args):
    """
    This is the generic function called by update_*_components for each graph type.
    It generates the list of input components for each graph.
    :param graph_type:
    :param component_list:
    :param dd_options:
    :param args:
    :return:
    """
    print('update_graph_components')
    # Generate the list of argument names based on the input order, paired by component id and property name
    args_dict = create_arguments_nested_dict(component_list, args)

    children = list()
    for component in component_list:
        name = component['id']
        # Pass most of the input arguments for this component to the constructor via
        # args_to_replicate. Remove component_type and label as they are used in other ways,
        # not passed to the constructor.
        args_to_replicate = dict(args_dict[name])
        del args_to_replicate['component_type']
        del args_to_replicate['label']
        del args_to_replicate['id']

        # Create a new instance of each component, with different constructors
        # for when the different types need different inputs
        if component['component_type'] == dcc.Dropdown:
            # Remove the options property to override it with the dd_options above
            del args_to_replicate['options']
            children.append([component['label'] + ":",
                             component['component_type'](
                                 id={'type': graph_type + '-' + name, 'index': args_dict[name]['id']['index']},
                                 **args_to_replicate,
                                 options=dd_options,
                                 )
                             ],
                            )
        else:
            children.append([component['label'] + ":",
                             component['component_type'](
                                 id={'type': graph_type + '-' + name, 'index': args_dict[name]['id']['index']},
                                 **args_to_replicate,
                                 )
                             ],
                            )

    return children


def generate_scatter_group(n_clicks):
    print('generate_scatter_group')
    return generate_generic_group(n_clicks, 'scatter', all_scatter_components)


def generate_bar_group(n_clicks):
    print('generate_bar_group')
    return generate_generic_group(n_clicks, 'bar', all_bar_components)


def generate_manhattan_group(n_clicks):
    print('generate_manhattan_group')
    return generate_generic_group(n_clicks, 'manhattan', all_manhattan_components)


def generate_generic_group(n_clicks, group_type, component_list):
    """
    The generic builder for each of the component types.
    :param n_clicks:
    :param group_type:
    :param component_list:
    :return:
    """
    print('generate_generic_group', group_type)
    children = list()

    for component in component_list:
        name = component['id']
        args_to_replicate = dict(component)
        del args_to_replicate['component_type']
        del args_to_replicate['id']
        del args_to_replicate['label']

        # Generate each component with the correct id, index, and arguments, inside its own Div.
        children.append(html.Div([component['label'] + ":",
                                  component['component_type'](id={'type': group_type + '-' + name, 'index': n_clicks},
                                                              **args_to_replicate,
                                                              )
                                  ],
                                 id={'type': 'div-' + group_type + '-' + name, 'index': n_clicks},
                                 style=style_dict
                                 )
                        )

    children.append(dcc.Graph(id={'type': 'gen-' + group_type + '-graph', 'index': n_clicks},
                              figure=go.Figure(data=go.Scatter()))
                    )
    print(children)

    return html.Div(id={'type': 'filter-graph-group-' + group_type,
                        'index': n_clicks
                        },
                    children=children
                    )


@app.callback(
    Output({'type': 'divgraph-type-dd', 'index': MATCH}, 'children'),
    [Input({'type': 'graph-type-dd', 'index': MATCH}, 'value')],
    [State({'type': 'graph-type-dd', 'index': MATCH}, 'id'),
     State({'type': 'divgraph-type-dd', 'index': MATCH}, 'children')]
)
def change_graph_group_type(graph_type, id, children):
    print('change_graph_group_type', graph_type, id)
    # Check whether the value of the dropdown matches the type of the existing group. If it doesn't match, then
    # generate a new group of the right type.
    if graph_type == 'Bar' and children[-1]['props']['id']['type'] != 'filter-graph-group-bar':
        children[-1] = generate_bar_group(id['index'])
    elif graph_type == 'Scatter' and children[-1]['props']['id']['type'] != 'filter-graph-group-scatter':
        children[-1] = generate_scatter_group(id['index'])
    elif graph_type == 'Manhattan' and children[-1]['props']['id']['type'] != 'filter-graph-group-manhattan':
        children[-1] = generate_manhattan_group(id['index'])
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
