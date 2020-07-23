import itertools
import json
import dash
import pandas as pd
import numpy as np
import dash_core_components as dcc
from dash.dependencies import Input, Output, State, MATCH
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from psych_dashboard.app import app, scatter_graph_dimensions, default_marker_color
from psych_dashboard.load_feather import load_filtered_feather


@app.callback(
    [Output({'type': 'div_scatter_'+str(t), 'index': MATCH}, 'children')
     for t in (list(scatter_graph_dimensions) + ['regression'])],
    [Input('df-loaded-div', 'children')],
    list(itertools.chain.from_iterable([State({'type': 'scatter_'+t, 'index': MATCH}, 'id'),
                                        State({'type': 'scatter_'+t, 'index': MATCH}, 'value')
                                        ] for t in (list(scatter_graph_dimensions) + ['regression'])))
    + [State({'type': 'div_scatter_x', 'index': MATCH}, 'style')]
)
def update_scatter_select_columns(df_loaded, x, xv, y, yv, color, colorv, size, sizev, facet_col, fcv, facet_row, frv,
                                  regression, regv, style_dict):
    print('update_scatter_select_columns')
    print(x, xv, y, yv, color, colorv, size, sizev, facet_col, fcv, facet_row, frv, regression, regv)
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
    dff = load_filtered_feather(df_loaded)
    options = [{'label': col,
                'value': col} for col in dff.columns]

    return ["x:", dcc.Dropdown(id={'type': 'scatter_x', 'index': x['index']}, options=options, value=xv)], \
           ["y:", dcc.Dropdown(id={'type': 'scatter_y', 'index': y['index']}, options=options, value=yv)], \
           ["color:", dcc.Dropdown(id={'type': 'scatter_color', 'index': color['index']},
                                   options=options, value=colorv)], \
           ["size:", dcc.Dropdown(id={'type': 'scatter_size', 'index': size['index']},
                                   options=options, value=sizev)], \
           ["split_horizontally:", dcc.Dropdown(id={'type': 'scatter_facet_col', 'index': facet_col['index']},
                                                options=options, value=fcv)], \
           ["split_vertically:", dcc.Dropdown(id={'type': 'scatter_facet_row', 'index': facet_row['index']},
                                              options=options, value=frv)], \
           ["regression degree:", dcc.Input(id={'type': 'scatter_regression', 'index': regression['index']},
                                            type='number',
                                            min=0,
                                            step=1,
                                            value=regv)]


def filter_facet(dff, facet, facet_cats, i):
    if facet is not None:
        return dff[dff[facet] == facet_cats[i]]

    return dff


min_marker_size = 2
max_marker_size = 10


@app.callback(
    Output({'type': 'gen_scatter_graph', 'index': MATCH}, "figure"),
    [*(Input({'type': 'scatter_'+d, 'index': MATCH}, "value") for d in (list(scatter_graph_dimensions) + ['regression']))],
    [State('df-loaded-div', 'children')]
)
def make_scatter_figure(x, y, color=None, size=None, facet_col=None, facet_row=None, regression_degree=None, df_loaded=None):
    print('make_scatter_figure')
    dff = load_filtered_feather(df_loaded)

    facet_row_cats = dff[facet_row].unique() if facet_row is not None else [None]
    facet_col_cats = dff[facet_col].unique() if facet_col is not None else [None]

    # Return empty scatter if not enough options are selected, or the data is empty.
    if dff.columns.size == 0 or x is None or y is None:
        return px.scatter()

    fig = make_subplots(len(facet_row_cats), len(facet_col_cats))
    # If color is not provided, then use default
    if color is None:
        color_to_use = default_marker_color
    # Otherwise, if the dtype is categorical, then we need to map - otherwise, leave as it is
    else:
        dff.dropna(inplace=True, subset=[color])
        color_to_use = pd.DataFrame(dff[color])

        if dff[color].dtype == pd.CategoricalDtype:
            dff['color_to_use'] = map_color(dff[color])
        else:
            color_to_use.set_index(dff.index, inplace=True)
            dff['color_to_use'] = color_to_use
    if size is not None:
        dff.dropna(inplace=True, subset=[size])
    for i in range(len(facet_row_cats)):
        for j in range(len(facet_col_cats)):
            working_dff = dff
            working_dff = filter_facet(working_dff, facet_row, facet_row_cats, i)
            working_dff = filter_facet(working_dff, facet_col, facet_col_cats, j)
            fig.add_trace(go.Scatter(x=working_dff[x],
                                     y=working_dff[y],
                                     mode='markers',
                                     marker=dict(color=working_dff[
                                         'color_to_use'] if 'color_to_use' in working_dff.columns else color_to_use,
                                                 coloraxis="coloraxis",
                                                 showscale=True),
                                     marker_size=map_size(working_dff[size], min_marker_size, max_marker_size) if size is not None else max_marker_size,
                                     hovertemplate=['SUBJECTKEY: ' + str(i) + '<br>EVENTNAME: ' + str(j) +
                                                    '<extra></extra>'
                                                    for (i, j) in working_dff.index],
                                     ),
                          i + 1,
                          j + 1)

            # Add regression lines
            print('regression_degree', regression_degree)
            if regression_degree is not None:
                working_dff.dropna(inplace=True)
                # Guard against fitting an empty graph
                if len(working_dff) > 0:
                    working_dff.sort_values(by=x, inplace=True)
                    Y = working_dff[y]
                    X = working_dff[x]
                    model = Pipeline([('poly', PolynomialFeatures(degree=regression_degree)),
                                      ('linear', LinearRegression(fit_intercept=False))])
                    reg = model.fit(np.vstack(X), Y)
                    Y_pred = reg.predict(np.vstack(X))
                    fig.add_trace(go.Scatter(name='line of best fit', x=X, y=Y_pred, mode='lines'))

    fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False, )
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')
    return fig


def map_color(dff):
    values = sorted(dff.unique())
    all_values = pd.Series(list(map(lambda x: values.index(x), dff)), index=dff.index)
    if all([value == 0 for value in all_values]):
        all_values = pd.Series([1 for _ in all_values])
    return all_values


def map_size(series, min_out, max_out):
    """Maps the range of series to [min_size, max_size]"""
    print('map_size', series)
    if series.empty:
        return []

    min_in = min(series)
    max_in = max(series)
    slope = 1.0 * (max_out - min_out) / (max_in - min_in)
    series = min_out + slope * (series - min_in)
    return series
