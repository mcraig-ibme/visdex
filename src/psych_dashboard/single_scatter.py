#
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline
# from psych_dashboard.app import app, graph_dimensions
# from psych_dashboard.load_feather import load_feather
#
# style_dict = {
#         'width': '13%',
#         'display': 'inline-block',
#         'verticalAlign': "middle",
#         'margin-right': '2em',
#     }
#
#
# @app.callback(
#     Output("graph1", "figure"),
#     [Input('df-loaded-div', 'children'),
#      *(Input(d, "value") for d in (list(graph_dimensions) + ['regression']))]
# )
# def make_scatter(df_loaded, x, y, color=None, facet_col=None, facet_row=None, regression_degree=None):
#     dff = load_feather(df_loaded)
#     print('make_scatter', x, y, color, facet_col, facet_row)
#
#     facet_row_cats = dff[facet_row].unique() if facet_row is not None else [None]
#     facet_col_cats = dff[facet_col].unique() if facet_col is not None else [None]
#
#     # Return empty scatter if not enough options are selected, or the data is empty.
#     if dff.columns.size == 0 or x is None or y is None:
#         return px.scatter()
#
#     fig = make_subplots(len(facet_row_cats), len(facet_col_cats))
#     # If color is not provided, then use default
#     if color is None:
#         color_to_use = default_marker_color
#     # Otherwise, if the dtype is categorical, then we need to map - otherwise, leave as it is
#     else:
#         dff.dropna(inplace=True, subset=[color])
#
#         color_to_use = pd.DataFrame(dff[color])
#
#         if dff[color].dtype == pd.CategoricalDtype:
#             dff['color_to_use'] = map_color(dff[color])
#         else:
#             color_to_use.set_index(dff.index, inplace=True)
#             dff['color_to_use'] = color_to_use
#     for i in range(len(facet_row_cats)):
#         for j in range(len(facet_col_cats)):
#             working_dff = dff
#             working_dff = filter_facet(working_dff, facet_row, facet_row_cats, i)
#             working_dff = filter_facet(working_dff, facet_col, facet_col_cats, j)
#
#             fig.add_trace(go.Scatter(x=working_dff[x],
#                                      y=working_dff[y],
#                                      mode='markers',
#                                      marker=dict(color=working_dff['color_to_use']
#                                                  if 'color_to_use' in working_dff.columns else color_to_use,
#                                                  coloraxis="coloraxis",
#                                                  showscale=True)
#                                      ),
#                           i + 1,
#                           j + 1)
#
#             # Add regression lines
#             if regression_degree is not None:
#                 working_dff.dropna(inplace=True)
#                 # Guard against fitting an empty graph
#                 if len(working_dff) > 0:
#                     working_dff.sort_values(by=x, inplace=True)
#                     Y = working_dff[y]
#                     X = working_dff[x]
#                     model = Pipeline([('poly', PolynomialFeatures(degree=regression_degree)),
#                                       ('linear', LinearRegression(fit_intercept=False))])
#                     reg = model.fit(np.vstack(X), Y)
#                     Y_pred = reg.predict(np.vstack(X))
#                     fig.add_trace(go.Scatter(name='line of best fit', x=X, y=Y_pred, mode='lines'))
#
#     fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False, )
#     fig.update_xaxes(matches='x')
#     fig.update_yaxes(matches='y')
#     return fig
#
#
# @app.callback(
#     dash.dependencies.Output('graph1-selection', 'children'),
#     [dash.dependencies.Input('df-loaded-div', 'children')])
# def update_select_columns(df_loaded):
#     print('update_select_columns')
#     dff = load_feather(df_loaded)
#     options = [{'label': col,
#                 'value': col} for col in dff.columns]
#     return html.Div([html.Div([value + ":", dcc.Dropdown(id=key, options=options)],
#                               style=style_dict
#                               )
#                      for key, value in graph_dimensions.items()
#                      ]
#
#                     + [html.Div(["regression:",
#                                  dcc.Input(id="regression",
#                                            type='number',
#                                            min=0,
#                                            step=1
#                                            )
#                                  ],
#                                 style=style_dict
#                                 )
#                        ]
#                     )
