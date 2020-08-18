import itertools
import json
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH
import plotly.graph_objects as go
from psych_dashboard.app import app, all_bar_components
from psych_dashboard.load_feather import load_filtered_feather


@app.callback(
    [Output({'type': 'div_bar_'+str(t), 'index': MATCH}, 'children')
     for t in [component['name'] for component in all_bar_components]],
    [Input('df-loaded-div', 'children')],
    [State({'type': 'div_bar_x', 'index': MATCH}, 'style')] +
    list(itertools.chain.from_iterable([State({'type': 'bar_'+t, 'index': MATCH}, 'id'),
                                        State({'type': 'bar_'+t, 'index': MATCH}, 'value')
                                        ] for t in [component['name'] for component in all_bar_components]))
)
def update_bar_select_columns(df_loaded, style_dict, *args):
    """ This function is triggered by a change to
    """
    print('update_bar_select_columns')
    keys = itertools.chain.from_iterable([str([component['name'] for component in all_bar_components][i]),
                                          str([component['name'] for component in all_bar_components][i])+'_val']
                                         for i in range(0, int(len(args)/2))
                                         )
    args_dict = dict(zip(keys, args))

    dff = load_filtered_feather()
    dd_options = [{'label': col,
                  'value': col} for col in dff.columns]

    children = list()
    for component in all_bar_components:
        name = component['name']
        if component['component_type'] == 'Dropdown':
            print(component, 'Dropdown')
            children.append([component['label'] + ":",
                                      dcc.Dropdown(id={'type': 'bar_' + str(name), 'index': args_dict[name]['index']},
                                                   options=dd_options)],
                                     )
        elif component['component_type'] == 'Input':
            print(component, 'Input')
            children.append([component['label'] + ":",
                                      dcc.Input(id={'type': 'bar_' + str(name), 'index': args_dict[name]['index']},
                                                type='number',
                                                min=0,
                                                step=1, )],
                                     )

    return children


@app.callback(
    Output({'type': 'gen_bar_graph', 'index': MATCH}, "figure"),
    [*(Input({'type': 'bar_'+d, 'index': MATCH}, "value") for d in [component['name'] for component in all_bar_components])],
)
def make_bar_figure(*args):
    print('make_bar_figure')
    keys = [str([component['name'] for component in all_bar_components][i]) for i in range(0, int(len(args)))]

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
