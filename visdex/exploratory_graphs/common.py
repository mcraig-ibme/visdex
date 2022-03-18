import logging
from collections import defaultdict

from dash.dependencies import Input, Output, State, MATCH
from dash import dcc

from visdex.data.cache import get_cache

LOG = logging.getLogger(__name__)

def common_define_cbs(app, ctype, make_fig):
    @app.callback(
        [
            Output({"type": f"div-{ctype}-{component['id']}", "index": MATCH}, "children")
            for component in all_components[ctype]
        ],
        [Input("df-loaded-div", "children")],
        [State({"type": f"div-{ctype}-x", "index": MATCH}, "style")]
        + [
            State({"type": f"{ctype}-{component['id']}", "index": MATCH}, prop)
            for component in all_components[ctype]
            for prop in component
        ],
    )
    def update_components(df_loaded, style_dict, *args):
        LOG.info(f"update_{ctype}_components")
        cache = get_cache()
        dff = cache.load("filtered")
        dd_options = [{"label": col, "value": col} for col in dff.columns]
        return update_graph_components(ctype, all_components[ctype], dd_options, args)

    @app.callback(
        Output({"type": f"gen-{ctype}-graph", "index": MATCH}, "figure"),
        [
            *(
                Input({"type": f"{ctype}-{component['id']}", "index": MATCH}, "value")
                for component in all_components[ctype]
            )
        ],
    )
    def make_figure(*args):
        LOG.info(f"make_{ctype}_figure")
        cache = get_cache()
        keys = [component["id"] for component in all_components[ctype]]

        args_dict = dict(zip(keys, args))
        dff = cache.load("filtered")
        return make_fig(dff, args_dict)

# Definitions of supported plot types
# All components should contain 'component_type', 'id', and 'label' as a minimum
all_components = dict(
    scatter=[
        {
            "component_type": dcc.Dropdown,
            "id": "x",
            "label": "x",
            "value": None,
            "multi": False,
            "options": [],
        },
        {
            "component_type": dcc.Dropdown,
            "id": "y",
            "label": "y",
            "value": None,
            "multi": False,
            "options": [],
        },
        {
            "component_type": dcc.Dropdown,
            "id": "color",
            "label": "color",
            "value": None,
            "multi": False,
            "options": [],
        },
        {
            "component_type": dcc.Dropdown,
            "id": "size",
            "label": "size",
            "value": None,
            "multi": False,
            "options": [],
        },
        {
            "component_type": dcc.Dropdown,
            "id": "facet_col",
            "label": "split horizontally",
            "value": None,
            "multi": False,
            "options": [],
        },
        {
            "component_type": dcc.Dropdown,
            "id": "facet_row",
            "label": "split vertically",
            "value": None,
            "multi": False,
            "options": [],
        },
        {
            "component_type": dcc.Input,
            "id": "regression",
            "label": "regression degree",
            "value": None,
            "type": "number",
            "min": 0,
            "step": 1,
        },
    ],
    bar=[
        {
            "component_type": dcc.Dropdown,
            "id": "x",
            "label": "x",
            "value": None,
            "options": [],
            "multi": False,
        },
        {
            "component_type": dcc.Dropdown,
            "id": "split_by",
            "label": "split by",
            "value": None,
            "options": [],
            "multi": False,
        },
    ],
    manhattan=[
        {
            "component_type": dcc.Dropdown,
            "id": "base_variable",
            "label": "base variable",
            "value": None,
            "options": [],
            "multi": False,
        },
        {
            "component_type": dcc.Input,
            "id": "pvalue",
            "label": "p-value",
            "value": 0.05,
            "type": "number",
            "min": 0,
            "step": 0.001,
        },
        {
            "component_type": dcc.Checklist,
            "id": "logscale",
            "label": "logscale",
            "options": [{"label": "", "value": "LOG"}],
            "value": ["LOG"],
        },
    ],
    violin=[
        {
            "component_type": dcc.Dropdown,
            "id": "base_variable",
            "label": "base variable",
            "value": None,
            "options": [],
            "multi": False,
        },
    ],
    histogram=[
        {
            "component_type": dcc.Dropdown,
            "id": "base_variable",
            "label": "base variable",
            "value": None,
            "options": [],
            "multi": False,
        },
        {
            "component_type": dcc.Input,
            "id": "nbins",
            "label": "n bins (1=auto)",
            "value": 1,
            "type": "number",
            "min": 1,
            "step": 1,
        },
    ],
)

def create_arguments_nested_dict(components_list, args):
    # Generate the list of argument names based on the input order, paired by component
    # id and property name
    keys = [
        (component["id"], prop) for component in components_list for prop in component
    ]
    # Convert inputs to a nested dict, with the outer key the component id, and the
    # inner key the property name
    args_dict = defaultdict(dict)
    for key, value in zip(keys, args):
        args_dict[key[0]][key[1]] = value
    return args_dict

def update_graph_components(graph_type, component_list, dd_options, args):
    """
    This is the generic function called by update_*_components for each graph type.
    It generates the list of input components for each graph.
    :param graph_type:
    :param component_list:
    :param dd_options:
    :param args:
    :return:
    """
    LOG.info(f"update_graph_components")
    # Generate the list of argument names based on the input order, paired by component
    # id and property name
    args_dict = create_arguments_nested_dict(component_list, args)

    children = list()
    for component in component_list:
        name = component["id"]
        # Pass most of the input arguments for this component to the constructor via
        # args_to_replicate. Remove component_type and label as they are used in other
        # ways, not passed to the constructor.
        args_to_replicate = dict(args_dict[name])
        del args_to_replicate["component_type"]
        del args_to_replicate["label"]
        del args_to_replicate["id"]

        # Create a new instance of each component, with different constructors
        # for when the different types need different inputs
        if component["component_type"] == dcc.Dropdown:
            # Remove the options property to override it with the dd_options above
            del args_to_replicate["options"]
            children.append(
                [
                    component["label"] + ":",
                    component["component_type"](
                        id={
                            "type": graph_type + "-" + name,
                            "index": args_dict[name]["id"]["index"],
                        },
                        **args_to_replicate,
                        options=dd_options,
                    ),
                ],
            )
        else:
            children.append(
                [
                    component["label"] + ":",
                    component["component_type"](
                        id={
                            "type": graph_type + "-" + name,
                            "index": args_dict[name]["id"]["index"],
                        },
                        **args_to_replicate,
                    ),
                ],
            )

    return children

