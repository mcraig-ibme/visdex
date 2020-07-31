import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, suppress_callback_exceptions=False, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

indices = ['SUBJECTKEY', 'EVENTNAME']
graph_types = ['Scatter', 'Bar']
dd_scatter_dims = {"x": "x",
                   "y": "y",
                   "color": "color (will drop NAs)",
                   "size": "size",
                   "facet_col": "split horizontally",
                   "facet_row": "split vertically"}
input_scatter_dims = {"regression": "regression degree"}
all_scatter_dims = {**dd_scatter_dims, **input_scatter_dims}

dd_bar_dims = {"x": "x",
               "split_by": "split by"}
input_bar_dims = {}
all_bar_dims = {**dd_bar_dims, **input_bar_dims}
default_marker_color = "crimson"
style_dict = {
        'width': '13%',
        'display': 'inline-block',
        'verticalAlign': "middle",
        'margin-right': '2em',
    }
