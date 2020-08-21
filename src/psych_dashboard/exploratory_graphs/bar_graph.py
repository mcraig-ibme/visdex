import dash_core_components as dcc
from dash.dependencies import Input, Output, State, MATCH
import plotly.graph_objects as go
from collections import defaultdict
from psych_dashboard.app import app, all_bar_components
from psych_dashboard.load_feather import load_filtered_feather


@app.callback(
    [Output({'type': 'div_bar_'+str(t), 'index': MATCH}, 'children')
     for t in [component['id'] for component in all_bar_components]],
    [Input('df-loaded-div', 'children')],
    [State({'type': 'div_bar_x', 'index': MATCH}, 'style')] +
    [State({'type': 'bar_' + component['id'], 'index': MATCH}, prop)
     for component in all_bar_components for prop in component]
)
def update_bar_select_columns(df_loaded, style_dict, *args):
    """ This function is triggered by a change to
    """
    print('update_bar_select_columns')
    # Generate the list of argument names based on the input order, paired by component id and property name
    keys = [(component['id'], str(prop))
            for component in all_bar_components for prop in component]
    # Convert inputs to a nested dict, with the outer key the component id, and the inner key the property name
    args_dict = defaultdict(dict)
    for key, value in zip(keys, args):
        args_dict[key[0]][key[1]] = value

    dff = load_filtered_feather()
    dd_options = [{'label': col,
                  'value': col} for col in dff.columns]

    children = list()
    for component in all_bar_components:
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
                                 id={'type': 'bar_' + name, 'index': args_dict[name]['id']['index']},
                                 **args_to_replicate,
                                 options=dd_options,
                                 )
                             ],
                            )
        else:
            children.append([component['label'] + ":",
                             component['component_type'](
                                 id={'type': 'bar_' + name, 'index': args_dict[name]['id']['index']},
                                 **args_to_replicate,
                                 )
                             ],
                            )

    print('children bar', children)
    return children


@app.callback(
    Output({'type': 'gen_bar_graph', 'index': MATCH}, "figure"),
    [*(Input({'type': 'bar_'+d, 'index': MATCH}, "value") for d in [component['id'] for component in all_bar_components])],
)
def make_bar_figure(*args):
    print('make_bar_figure')
    keys = [str([component['id'] for component in all_bar_components][i]) for i in range(0, int(len(args)))]

    args_dict = dict(zip(keys, args))
    dff = load_filtered_feather()

    # Return empty scatter if not enough options are selected, or the data is empty.
    if dff.columns.size == 0 or args_dict['x'] is None:
        return go.Figure(go.Bar())

    fig = go.Figure()

    if args_dict['split_by'] is not None:
        # get all unique values in split_by column.
        # Filter by each of these, and add a go.Bar for each, and use these as names.
        split_by_names = sorted(dff[args_dict['split_by']].dropna().unique())

        for name in split_by_names:
            count_by_value = dff[dff[args_dict['split_by']] == name][args_dict['x']].value_counts()

            fig.add_trace(go.Bar(name=name, x=count_by_value.index, y=count_by_value.values))
    else:

        count_by_value = dff[args_dict['x']].value_counts()

        fig.add_trace(go.Bar(name=str(args_dict['x']), x=count_by_value.index, y=count_by_value.values))
    fig.update_layout(coloraxis=dict(colorscale='Bluered_r'))

    return fig
