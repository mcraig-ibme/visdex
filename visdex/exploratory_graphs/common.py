import logging
from collections import defaultdict

import dash_core_components as dcc

LOG = logging.getLogger(__name__)

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

