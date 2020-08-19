import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, suppress_callback_exceptions=False, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

indices = ['SUBJECTKEY', 'EVENTNAME']
graph_types = ['Scatter', 'Bar', 'Manhattan']

all_scatter_components = [{'component_type': 'Dropdown',
                     'id': 'x',
                     'label': 'x',
                     'starting_value': None,
                     'other_args': {'multi': False,
                                    'starting_options': []}
                     },
                    {'component_type': 'Dropdown',
                     'id': 'y',
                     'label': 'y',
                     'starting_value': None,
                     'other_args': {'multi': False,
                                    'starting_options': []}
                     },
                    {'component_type': 'Dropdown',
                     'id': 'color',
                     'label': 'color',
                     'starting_value': None,
                     'other_args': {'multi': False,
                                    'starting_options': []}
                     },
                    {'component_type': 'Dropdown',
                     'id': 'size',
                     'label': 'size',
                     'starting_value': None,
                     'other_args': {'multi': False,
                                    'starting_options': []}
                     },
                    {'component_type': 'Dropdown',
                     'id': 'facet_col',
                     'label': 'split horizontally',
                     'starting_value': None,
                     'other_args': {'multi': False,
                                    'starting_options': []}
                     },
                    {'component_type': 'Dropdown',
                     'id': 'facet_row',
                     'label': 'split vertically',
                     'starting_value': None,
                     'other_args': {'multi': False,
                                    'starting_options': []}
                     },
                    {'component_type': 'Input',
                     'id': 'regression',
                     'label': 'regression degree',
                     'starting_value': None,
                     'other_args': {'type': 'number',
                                    'min': 0,
                                    'step': 1}
                     },
                    ]

all_bar_components = [{'component_type': 'Dropdown',
                       'id': 'x',
                       'label': 'x',
                       'starting_value': None,
                       'starting_options': [],
                       'other_args': {'multi': False,
                                      'starting_options': []}},
                      {'component_type': 'Dropdown',
                       'id': 'split_by',
                       'label': 'split by',
                       'starting_value': None,
                       'other_args': {'multi': False,
                                      'starting_options': []}},
                      ]

dd_manhattan_dims = {"base_variable": "base variable"}
input_manhattan_dims = {"pvalue": "p-value"}
check_manhattan_dims = {"logscale": "logscale"}
all_manhattan_components = [{'component_type': 'Dropdown',
                       'id': 'base_variable',
                       'label': 'base variable',
                       'starting_value': None,
                       'starting_options': [],
                       'other_args': {'multi': False,
                                      'starting_options': []}},
                      {'component_type': 'Input',
                       'id': 'pvalue',
                       'label': 'p-value',
                       'starting_value': 0.05,
                       'other_args': {'type': 'number',
                                      'min': 0,
                                      'step': 0.001}
                       },
                      {'component_type': 'Checklist',
                       'id': 'logscale',
                       'label': 'logscale',
                       'other_args': {'options': [{'label': '', 'value': 'LOG'}],
                                      'value': ['LOG']}
                       },
                      ]

default_marker_color = "crimson"
style_dict = {
        'width': '13%',
        'display': 'inline-block',
        'verticalAlign': "middle",
        'margin-right': '2em',
    }
