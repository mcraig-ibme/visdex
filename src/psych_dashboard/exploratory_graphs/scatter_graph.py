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
from psych_dashboard.app import app, dd_scatter_dims, input_scatter_dims, all_scatter_dims, default_marker_color
from psych_dashboard.load_feather import load_filtered_feather


@app.callback(
    [Output({'type': 'div_scatter_'+str(t), 'index': MATCH}, 'children')
     for t in list(all_scatter_dims)],
    [Input('df-loaded-div', 'children')],
    [State({'type': 'div_scatter_x', 'index': MATCH}, 'style')] +
    list(itertools.chain.from_iterable([State({'type': 'scatter_'+t, 'index': MATCH}, 'id'),
                                        State({'type': 'scatter_'+t, 'index': MATCH}, 'value')
                                        ] for t in list(all_scatter_dims)))
)
def update_scatter_select_columns(df_loaded, style_dict, *args):
    print('update_scatter_select_columns')
    # Generate the list of argument names based on the input order
    keys = itertools.chain.from_iterable([str(list(all_scatter_dims.keys())[i]),
                                          str(list(all_scatter_dims.keys())[i])+'_val']
                                         for i in range(0, int(len(args)/2))
                                         )

    # Convert inputs to a dict called 'args_dict'
    args_dict = dict(zip(keys, args))

    dff = load_filtered_feather()
    dd_options = [{'label': col,
                   'value': col} for col in dff.columns]

    return tuple([[dd_scatter_dims[dim] + ":",
                   dcc.Dropdown(id={'type': 'scatter_'+dim, 'index': args_dict[dim]['index']},
                                options=dd_options,
                                value=args_dict[dim+'_val'])
                   ] for dim in dd_scatter_dims.keys()] +
                 [[input_scatter_dims[dim] + ":",
                   dcc.Input(id={'type': 'scatter_'+dim, 'index': args_dict[dim]['index']},
                             type='number',
                             min=0,
                             step=1,
                             value=args_dict[dim+'_val'])
                   ] for dim in input_scatter_dims.keys()]
                 )


def make_subplot_titles(facet_row, facet_row_cats, facet_col, facet_col_cats):
    """
    Combine the supplied name of the row and column facets, and the categories detected within them, to create labels
    for each of the subplots.
    If facets are used in both directions, then format is
    SEX=M, AGE=12
    whereas if facets are used in only one direction, it is
    SEX=M.
    None is returned if no facets in use.
    """
    if facet_row is not None and facet_col is not None:
        return [str(facet_row) + '=' + str(row) + ', ' + str(facet_col) + '=' + str(col)
                for row, col in itertools.product(facet_row_cats, facet_col_cats)]
    elif facet_row is not None:
        return [str(facet_row) + '=' + str(row) for row in facet_row_cats]
    elif facet_col is not None:
        return [str(facet_col) + '=' + str(col) for col in facet_col_cats]
    else:
        return None


def filter_facet(dff, facet, facet_cats, i):
    if facet is not None:
        return dff[dff[facet] == facet_cats[i]]

    return dff


min_marker_size = 2
max_marker_size = 10


@app.callback(
    Output({'type': 'gen_scatter_graph', 'index': MATCH}, "figure"),
    [*(Input({'type': 'scatter_'+d, 'index': MATCH}, "value") for d in all_scatter_dims)],
)
def make_scatter_figure(*args):
    print('make_scatter_figure')
    # Generate the list of argument names based on the input order
    keys = [str(list(all_scatter_dims.keys())[i]) for i in range(0, int(len(args)))]

    # Convert inputs to a dict called 'args_dict'
    args_dict = dict(zip(keys, args))
    dff = load_filtered_feather()

    facet_row_cats = list(dff[args_dict['facet_row']].unique()) if args_dict['facet_row'] is not None else [None]
    facet_col_cats = list(dff[args_dict['facet_col']].unique()) if args_dict['facet_col'] is not None else [None]

    try:
        if len(facet_row_cats) > 1:
            facet_row_cats.remove(None)
    except ValueError:
        pass
    try:
        if len(facet_col_cats) > 1:
            facet_col_cats.remove(None)
    except ValueError:
        pass

    # Return empty scatter if not enough options are selected, or the data is empty.
    if dff.columns.size == 0 or args_dict['x'] is None or args_dict['y'] is None:
        return px.scatter()

    # Create titles for each of the subplots, and initialise the subplots with them.
    subplot_titles = make_subplot_titles(args_dict['facet_row'], facet_row_cats, args_dict['facet_col'], facet_col_cats)
    fig = make_subplots(rows=len(facet_row_cats), cols=len(facet_col_cats),
                        subplot_titles=subplot_titles)
    
    # If color is not provided, then use default
    if args_dict['color'] is None:
        color_to_use = default_marker_color
    # Otherwise, if the dtype is categorical, then we need to map - otherwise, leave as it is
    else:
        dff.dropna(inplace=True, subset=[args_dict['color']])
        color_to_use = pd.DataFrame(dff[args_dict['color']])

        if args_dict['color'] in dff.select_dtypes(include='object').columns:
            dff['color_to_use'] = map_color(dff[args_dict['color']])
        else:
            color_to_use.set_index(dff.index, inplace=True)
            dff['color_to_use'] = color_to_use
    if args_dict['size'] is not None:
        dff.dropna(inplace=True, subset=[args_dict['size']])
    for i in range(len(facet_row_cats)):
        for j in range(len(facet_col_cats)):
            working_dff = dff
            working_dff = filter_facet(working_dff, args_dict['facet_row'], facet_row_cats, i)
            working_dff = filter_facet(working_dff, args_dict['facet_col'], facet_col_cats, j)

            fig.add_trace(go.Scatter(x=working_dff[args_dict['x']],
                                     y=working_dff[args_dict['y']],
                                     mode='markers',
                                     marker=dict(color=working_dff[
                                         'color_to_use'] if 'color_to_use' in working_dff.columns else color_to_use,
                                                 coloraxis="coloraxis",
                                                 showscale=True),
                                     marker_size=map_size(working_dff[args_dict['size']], min_marker_size, max_marker_size) if args_dict['size'] is not None else max_marker_size,
                                     hovertemplate=['SUBJECTKEY: ' + str(i) + '<br>EVENTNAME: ' + str(j) +
                                                    '<extra></extra>'
                                                    for (i, j) in working_dff.index],
                                     ),
                          row=i + 1,
                          col=j + 1)

            # Add regression lines
            print('regression', args_dict['regression'])
            if args_dict['regression'] is not None:
                working_dff.dropna(inplace=True)
                # Guard against fitting an empty graph
                if len(working_dff) > 0:
                    working_dff.sort_values(by=args_dict['x'], inplace=True)
                    Y = working_dff[args_dict['y']]
                    X = working_dff[args_dict['x']]
                    model = Pipeline([('poly', PolynomialFeatures(degree=args_dict['regression'])),
                                      ('linear', LinearRegression(fit_intercept=False))])
                    reg = model.fit(np.vstack(X), Y)
                    Y_pred = reg.predict(np.vstack(X))
                    fig.add_trace(go.Scatter(name='line of best fit', x=X, y=Y_pred, mode='lines'),
                                  row=i + 1,
                                  col=j + 1)

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
