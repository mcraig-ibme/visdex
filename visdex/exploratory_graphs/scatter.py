"""
visdex: Scatter graph
"""
import logging
import itertools

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

LOG = logging.getLogger(__name__)

MIN_MARKER_SIZE = 2
MAX_MARKER_SIZE = 10
DEFAULT_MARKER_COLOR = "crimson"

def make_figure(dff, args_dict):
    facet_row_cats = (
        list(dff[args_dict["facet_row"]].unique())
        if args_dict["facet_row"] is not None
        else [None]
    )
    facet_col_cats = (
        list(dff[args_dict["facet_col"]].unique())
        if args_dict["facet_col"] is not None
        else [None]
    )

    # Remove nans and Nones
    if len(facet_row_cats) > 1:
        try:
            facet_row_cats = [x for x in facet_row_cats if ~np.isnan(x)]
        except TypeError:
            pass
        try:
            facet_row_cats.remove(None)
        except ValueError:
            pass
    if len(facet_col_cats) > 1:
        try:
            facet_col_cats = [x for x in facet_col_cats if ~np.isnan(x)]
        except TypeError:
            pass
        try:
            facet_col_cats.remove(None)
        except ValueError:
            pass

    # Return empty scatter if not enough options are selected, or the data is empty.
    if dff.columns.size == 0 or args_dict["x"] is None or args_dict["y"] is None:
        return px.scatter()

    # Create titles for each of the subplots, and initialise the subplots with them.
    subplot_titles = make_subplot_titles(
        args_dict["facet_row"], facet_row_cats, args_dict["facet_col"], facet_col_cats
    )
    fig = make_subplots(
        rows=len(facet_row_cats),
        cols=len(facet_col_cats),
        subplot_titles=subplot_titles,
    )

    # If color is not provided, then use default
    if args_dict["color"] is None:
        color_to_use = DEFAULT_MARKER_COLOR
    # Otherwise, if the dtype is categorical, then we need to map - otherwise, leave as it is
    else:
        dff.dropna(inplace=True, subset=[args_dict["color"]])
        color_to_use = pd.DataFrame(dff[args_dict["color"]])

        if args_dict["color"] in dff.select_dtypes(include="object").columns:
            dff["color_to_use"] = map_color(dff[args_dict["color"]])
        else:
            color_to_use.set_index(dff.index, inplace=True)
            dff["color_to_use"] = color_to_use
    if args_dict["size"] is not None:
        dff.dropna(inplace=True, subset=[args_dict["size"]])

    annotations = []
    for i in range(len(facet_row_cats)):
        for j in range(len(facet_col_cats)):
            working_dff = dff
            working_dff = filter_facet(
                working_dff, args_dict["facet_row"], facet_row_cats, i
            )
            working_dff = filter_facet(
                working_dff, args_dict["facet_col"], facet_col_cats, j
            )

            fig.add_trace(
                go.Scatter(
                    x=working_dff[args_dict["x"]],
                    y=working_dff[args_dict["y"]],
                    mode="markers",
                    marker=dict(
                        color=working_dff["color_to_use"]
                        if "color_to_use" in working_dff.columns
                        else color_to_use,
                        coloraxis="coloraxis",
                        showscale=True,
                    ),
                    marker_size=map_size(
                        working_dff[args_dict["size"]], MIN_MARKER_SIZE, MAX_MARKER_SIZE
                    )
                    if args_dict["size"] is not None
                    else MAX_MARKER_SIZE,
                    # FIXME tied to exact form of index
                    hovertemplate=[str(v) for v in working_dff.index]
                    #hovertemplate=[
                    #    "SUBJECTKEY: "
                    #    + str(i)
                    #    + "<br>EVENTNAME: "
                    #    + str(j)
                    #    + "<extra></extra>"
                    #    for (i, j) in working_dff.index
                    #],
                ),
                row=i + 1,
                col=j + 1,
            )

            # Add regression lines
            if args_dict["regression"] is not None:
                working_dff.dropna(inplace=True)
                # Guard against fitting an empty graph
                if len(working_dff) > 0:
                    working_dff.sort_values(by=args_dict["x"], inplace=True)
                    Y = working_dff[args_dict["y"]]
                    X = working_dff[args_dict["x"]]
                    model = Pipeline(
                        [
                            (
                                "poly",
                                PolynomialFeatures(degree=args_dict["regression"]),
                            ),
                            ("linear", LinearRegression(fit_intercept=False)),
                        ]
                    )
                    try:
                        reg = model.fit(np.vstack(X), Y)
                        Y_pred = reg.predict(np.vstack(X))
                        r2 = r2_score(np.vstack(Y), Y_pred)
                        LOG.debug(f"r2 is {r2}")
                        fig.add_trace(
                            go.Scatter(
                                name="line of best fit", x=X, y=Y_pred, mode="lines"
                            ),
                            row=i + 1,
                            col=j + 1,
                        )
                        annotations.append(
                            dict(
                                x=working_dff[args_dict["x"]].max(),
                                y=working_dff[args_dict["y"]].max(),
                                xref="x" + str(i + 1 + j * len(facet_row_cats)),
                                yref="y" + str(i + 1 + j * len(facet_row_cats)),
                                xanchor="left",
                                yanchor="bottom",
                                text=f"r^2 = {r2:.2f}",
                                showarrow=False,
                                ax=0,
                                ay=0,
                            )
                        )
                    except (TypeError, ValueError) as e:
                        LOG.debug(e)
                        pass

    for annotation in annotations:
        fig.add_annotation(annotation)

    # Generate the appropriate title
    title = f"Scatter plot(s) of {args_dict['x']} against {args_dict['y']}"
    if args_dict["facet_row"] is not None and args_dict["facet_col"] is not None:
        title += f", split by {args_dict['facet_row']} and {args_dict['facet_col']}"
    elif args_dict["facet_row"] is not None:
        title += f", split by {args_dict['facet_row']}"
    elif args_dict["facet_col"] is not None:
        title += f", split by {args_dict['facet_col']}"

    if args_dict["color"] is not None and args_dict["size"] is not None:
        title += f", coloured by {args_dict['color']} and sized by {args_dict['size']}"
    elif args_dict["color"] is not None:
        title += f", coloured by {args_dict['color']}"
    elif args_dict["size"] is not None:
        title += f", sized by {args_dict['size']}"

    fig.update_layout(
        coloraxis=dict(colorscale="Bluered_r"),
        showlegend=False,
        title=title,
    )
    fig.update_xaxes(matches="x")
    fig.update_yaxes(matches="y")
    return fig

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
        return [
            str(facet_row) + "=" + str(row) + ", " + str(facet_col) + "=" + str(col)
            for row, col in itertools.product(facet_row_cats, facet_col_cats)
        ]
    elif facet_row is not None:
        return [str(facet_row) + "=" + str(row) for row in facet_row_cats]
    elif facet_col is not None:
        return [str(facet_col) + "=" + str(col) for col in facet_col_cats]
    else:
        return None

def filter_facet(dff, facet, facet_cats, i):
    if facet is not None:
        return dff[dff[facet] == facet_cats[i]]

    return dff

def map_color(dff):
    values = sorted(dff.unique())
    all_values = pd.Series(list(map(lambda x: values.index(x), dff)), index=dff.index)
    if all([value == 0 for value in all_values]):
        all_values = pd.Series([1 for _ in all_values])
    return all_values


def map_size(series, min_out, max_out):
    """Maps the range of series to [min_size, max_size]"""
    if series.empty:
        return []

    min_in = min(series)
    max_in = max(series)
    slope = 1.0 * (max_out - min_out) / (max_in - min_in)
    series = min_out + slope * (series - min_in)
    return series
