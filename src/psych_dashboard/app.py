import dash

app = dash.Dash(__name__, suppress_callback_exceptions=False)
server = app.server

indices = ['SUBJECTKEY', 'EVENTNAME']
graph_types = ['Scatter', 'Bar']
scatter_graph_dimensions = {"x": "x",
                            "y": "y",
                            "color": "color (will drop NAs)",
                            "facet_col": "split horizontally",
                            "facet_row": "split vertically"}
bar_graph_dimensions = {"x": "x",
                        "split_by": "split by"}
