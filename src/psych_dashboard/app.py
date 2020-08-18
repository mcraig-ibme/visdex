import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, suppress_callback_exceptions=False, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

indices = ['SUBJECTKEY', 'EVENTNAME']
graph_types = ['Scatter', 'Bar', 'Manhattan']

all_scatter_components = [{'component_type': 'Dropdown',
                     'name': 'x',
                     'label': 'x',
                     'starting_value': None,
                     'other_args': {'multi': False,
                                    'starting_options': []}
                     },
                    {'component_type': 'Dropdown',
                     'name': 'y',
                     'label': 'y',
                     'starting_value': None,
                     'other_args': {'multi': False,
                                    'starting_options': []}
                     },
                    {'component_type': 'Dropdown',
                     'name': 'color',
                     'label': 'color',
                     'starting_value': None,
                     'other_args': {'multi': False,
                                    'starting_options': []}
                     },
                    {'component_type': 'Dropdown',
                     'name': 'size',
                     'label': 'size',
                     'starting_value': None,
                     'other_args': {'multi': False,
                                    'starting_options': []}
                     },
                    {'component_type': 'Dropdown',
                     'name': 'facet_col',
                     'label': 'split horizontally',
                     'starting_value': None,
                     'other_args': {'multi': False,
                                    'starting_options': []}
                     },
                    {'component_type': 'Dropdown',
                     'name': 'facet_row',
                     'label': 'split vertically',
                     'starting_value': None,
                     'other_args': {'multi': False,
                                    'starting_options': []}
                     },
                    {'component_type': 'Input',
                     'name': 'regression',
                     'label': 'regression degree',
                     'starting_value': None,
                     'other_args': {'type': 'number',
                                    'min': 0,
                                    'step': 1}
                     },
                    ]

all_bar_components = [{'component_type': 'Dropdown',
                       'name': 'x',
                       'label': 'x',
                       'starting_value': None,
                       'starting_options': [],
                       'other_args': {'multi': False,
                                      'starting_options': []}},
                      {'component_type': 'Dropdown',
                       'name': 'split_by',
                       'label': 'split by',
                       'starting_value': None,
                       'other_args': {'multi': False,
                                      'starting_options': []}},
                      ]

dd_manhattan_dims = {"base_variable": "base variable"}
input_manhattan_dims = {"pvalue": "p-value"}
check_manhattan_dims = {"logscale": "logscale"}
all_manhattan_dims = {**dd_manhattan_dims, **input_manhattan_dims, **check_manhattan_dims}

default_marker_color = "crimson"
style_dict = {
        'width': '13%',
        'display': 'inline-block',
        'verticalAlign': "middle",
        'margin-right': '2em',
    }
