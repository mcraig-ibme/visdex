import itertools
import json
import dash
import dash_core_components as dcc
from dash.dependencies import Input, Output, State, MATCH
import plotly.graph_objects as go
from psych_dashboard.app import app, bar_graph_dimensions
from psych_dashboard.load_feather import load_feather


@app.callback(
    [Output({'type': 'div_bar_'+str(t), 'index': MATCH}, 'children')
     for t in bar_graph_dimensions],
    [Input('df-loaded-div', 'children')],
    list(itertools.chain.from_iterable([State({'type': 'bar_'+t, 'index': MATCH}, 'id'),
                                        State({'type': 'bar_'+t, 'index': MATCH}, 'value')
                                        ] for t in bar_graph_dimensions))
    + [State({'type': 'div_bar_x', 'index': MATCH}, 'style')]
)
def update_bar_select_columns(df_loaded, x, xv, split_by, split_byv,
                              style_dict):
    print('update_bar_select_columns')
    print(x, xv, split_by, split_byv)
    print('style_dict')
    print(style_dict)
    ctx = dash.callback_context
    ctx_msg = json.dumps(
        {
            'states': ctx.states,
            'triggered': ctx.triggered
        },
        indent=2)
    print(ctx_msg)
    dff = load_feather(df_loaded)
    options = [{'label': col,
                'value': col} for col in dff.columns]

    return ["x:", dcc.Dropdown(id={'type': 'bar_x', 'index': x['index']}, options=options, value=xv)], \
           ["split by:", dcc.Dropdown(id={'type': 'bar_split_by', 'index': split_by['index']},
                                      options=options, value=split_byv)],


@app.callback(
    Output({'type': 'gen_bar_graph', 'index': MATCH}, "figure"),
    [*(Input({'type': 'bar_'+d, 'index': MATCH}, "value") for d in bar_graph_dimensions)],
    [State('df-loaded-div', 'children')]

)
def make_bar_figure(x, split_by=None, df_loaded=None):
    dff = load_feather(df_loaded)
    print('make_bar_figure', x, split_by)
    ctx = dash.callback_context
    ctx_msg = json.dumps({
        'states': ctx.states,
        'triggered': ctx.triggered
    }, indent=2)
    print(ctx_msg)

    # Return empty scatter if not enough options are selected, or the data is empty.
    if dff.columns.size == 0 or x is None:
        return go.Figure(go.Bar())

    fig = go.Figure()

    if split_by is not None:
        # get all unique values in split_by column.
        # Filter by each of these, and add a go.Bar for each, and use these as names.
        split_by_names = sorted(dff[split_by].dropna().unique())

        for name in split_by_names:
            count_by_value = dff[dff[split_by] == name][x].value_counts()

            fig.add_trace(go.Bar(name=name, x=count_by_value.index, y=count_by_value.values))
    else:

        count_by_value = dff[x].value_counts()

        fig.add_trace(go.Bar(name=str(x), x=count_by_value.index, y=count_by_value.values))
    fig.update_layout(coloraxis=dict(colorscale='Bluered_r'))

    return fig