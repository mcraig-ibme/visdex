import os
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from flask_caching import Cache

app = dash.Dash(__name__, suppress_callback_exceptions=False, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

CACHE_CONFIG = {
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379')
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)


indices = ['SUBJECTKEY', 'EVENTNAME']
graph_types = ['Scatter', 'Bar', 'Manhattan']

standard_margin_left = '10px'
div_style = {'margin-left': standard_margin_left}


# All components should contain 'component_type', 'id', and 'label' as a minimum
all_scatter_components = [{'component_type': dcc.Dropdown,
                           'id': 'x',
                           'label': 'x',
                           'value': None,
                           'multi': False,
                           'options': [],
                           },
                          {'component_type': dcc.Dropdown,
                           'id': 'y',
                           'label': 'y',
                           'value': None,
                           'multi': False,
                           'options': [],
                           },
                          {'component_type': dcc.Dropdown,
                           'id': 'color',
                           'label': 'color',
                           'value': None,
                           'multi': False,
                           'options': [],
                           },
                          {'component_type': dcc.Dropdown,
                           'id': 'size',
                           'label': 'size',
                           'value': None,
                           'multi': False,
                           'options': [],
                           },
                          {'component_type': dcc.Dropdown,
                           'id': 'facet_col',
                           'label': 'split horizontally',
                           'value': None,
                           'multi': False,
                           'options': [],
                           },
                          {'component_type': dcc.Dropdown,
                           'id': 'facet_row',
                           'label': 'split vertically',
                           'value': None,
                           'multi': False,
                           'options': [],
                           },
                          {'component_type': dcc.Input,
                           'id': 'regression',
                           'label': 'regression degree',
                           'value': None,
                           'type': 'number',
                           'min': 0,
                           'step': 1,
                           },
                          ]

all_bar_components = [{'component_type': dcc.Dropdown,
                       'id': 'x',
                       'label': 'x',
                       'value': None,
                       'options': [],
                       'multi': False,
                       },
                      {'component_type': dcc.Dropdown,
                       'id': 'split_by',
                       'label': 'split by',
                       'value': None,
                       'options': [],
                       'multi': False,
                       },
                      ]

all_manhattan_components = [{'component_type': dcc.Dropdown,
                             'id': 'base_variable',
                             'label': 'base variable',
                             'value': None,
                             'options': [],
                             'multi': False
                             },
                            {'component_type': dcc.Input,
                             'id': 'pvalue',
                             'label': 'p-value',
                             'value': 0.05,
                             'type': 'number',
                             'min': 0,
                             'step': 0.001
                             },
                            {'component_type': dcc.Checklist,
                             'id': 'logscale',
                             'label': 'logscale',
                             'options': [{'label': '', 'value': 'LOG'}],
                             'value': ['LOG']
                             },
                            ]

default_marker_color = "crimson"
style_dict = {
        'width': '13%',
        'display': 'inline-block',
        'verticalAlign': "middle",
        'margin-right': '2em',
    }
