import itertools
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from sklearn.cluster import AgglomerativeClustering
from psych_dashboard.app import app
from psych_dashboard.load_feather import store, load
from psych_dashboard.timing import timing, start_timer, log_timing, print_timings

logging.getLogger(__name__)


@app.callback(
    [Output("heatmap-dropdown", "options"), Output("heatmap-dropdown", "value")],
    [Input("df-filtered-loaded-div", "children")],
    prevent_initial_call=True,
)
@timing
def update_heatmap_dropdown(df_loaded):
    logging.info(f"update_heatmap_dropdown {df_loaded}")
    dff = load("filtered")

    options = [
        {"label": col, "value": col}
        for col in dff.columns
        if dff[col].dtype in [np.int64, np.float64]
    ]
    return options, [
        col for col in dff.columns if dff[col].dtype in [np.int64, np.float64]
    ]


def flattened(df):
    """
    Convert a DF into a Series, where the MultiIndex of each element is a combination
    of the index/col from the original DF
    """
    # The series contains only half of the matrix, so filter by the order of the two
    # level labels.
    s = pd.Series(
        index=pd.MultiIndex.from_tuples(
            filter(
                lambda x: df.index.get_loc(x[0]) < df.index.get_loc(x[1]),
                list(itertools.product(df.index, df.columns)),
            ),
            names=["first", "second"],
        ),
        name="value",
    )

    for (a, b) in s.index:
        s[a, b] = df[b][a]

    return s


def reorder_df(df, order):
    """
    Change the row and column order of df to that given in order
    """
    return df.reindex(order)[order]


def recalculate_corr_etc(selected_columns, dff, corr_dff, pval_dff, logs_dff):
    start_timer("recalculate_corr_etc")
    # Work out which columns/rows are needed anew, and which are already populated
    # TODO: note that if we load in a new file with some of the same column names,
    #  then this old correlation data may be used erroneously.
    existing_cols = corr_dff.columns
    overlap = list(set(selected_columns).intersection(set(existing_cols)))
    logging.debug(f"these are needed and already available: {overlap}")
    required_new = list(set(selected_columns).difference(set(existing_cols)))
    logging.debug(f"these are needed and not already available: {required_new}")

    ########
    # Create initial existing vs existing DF
    ########

    # If there is overlap, then create brand new empty dataframes.
    # Otherwise, update the existing dataframes.
    if len(overlap) == 0:
        logging.debug(f"create new")
        corr = pd.DataFrame()
        pvalues = pd.DataFrame()
        logs = pd.DataFrame()

    else:
        # Copy across existing data rather than recalculating (so in this operation
        # we drop the unneeded elements)
        # Then create nan elements in corr, p-values and logs matrices for those values
        # which will be calculated.
        corr = corr_dff.loc[overlap, overlap]
        pvalues = pval_dff.loc[overlap, overlap]
        logs = logs_dff.loc[overlap, overlap]

    log_timing("recalculate_corr_etc", "update_summary_heatmap-init-corr")

    # Populate missing elements in correlation matrix and p-values matrix using
    # stats.pearsonr
    # Firstly, convert the dff columns needed to numpy (far faster
    # than doing it each iteration)
    start_timer("inner")

    np_overlap = dff[overlap].to_numpy()
    np_req = dff[required_new].to_numpy()

    log_timing("inner", "update_summary_heatmap-numpy")  # This is negligible

    ########
    # Create new vs existing NumPy arrays, fill with calculated data. Then convert to
    # DFs, and append those to existing vs existing DF, to create all vs existing DFs
    ########
    new_against_existing_corr = np.full(
        shape=[len(overlap), len(required_new)], fill_value=np.nan
    )
    new_against_existing_pval = np.full(
        shape=[len(overlap), len(required_new)], fill_value=np.nan
    )
    new_against_existing_logs = np.full(
        shape=[len(overlap), len(required_new)], fill_value=np.nan
    )

    log_timing("inner", "update_summary_heatmap-nae_init")

    for v2 in range(len(required_new)):
        for v1 in range(len(overlap)):
            # Mask out any pairs that contain nans (this is done pairwise rather than
            # using .dropna on the full dataframe)
            mask = ~np.isnan(np_overlap[:, v1]) & ~np.isnan(np_req[:, v2])
            c, p = stats.pearsonr(np_overlap[mask, v1], np_req[mask, v2])
            new_against_existing_corr[v1, v2] = c
            new_against_existing_pval[v1, v2] = p
            new_against_existing_logs[v1, v2] = -np.log10(p)

    log_timing("inner", "update_summary_heatmap-nae_calc")

    new_against_existing_corr_df = pd.DataFrame(
        data=new_against_existing_corr, columns=required_new, index=overlap
    )
    corr[required_new] = new_against_existing_corr_df
    # logging.debug(f'corr {corr}')
    new_against_existing_pval_df = pd.DataFrame(
        data=new_against_existing_pval, columns=required_new, index=overlap
    )
    pvalues[required_new] = new_against_existing_pval_df
    # As new_against_existing_logs doesn't need to be transposed (the transpose is
    # nans instead), don't use an intermediate DF.
    logs[required_new] = pd.DataFrame(
        data=new_against_existing_logs, columns=required_new, index=overlap
    )

    log_timing("inner", "update_summary_heatmap-nae_copy")

    ########
    # Create existing vs new DFs by transpose (apart from logs, whose transpose is nans)
    ########

    existing_against_new_corr = new_against_existing_corr_df.transpose()
    existing_against_new_pval = new_against_existing_pval_df.transpose()
    existing_against_new_logs = pd.DataFrame(
        data=np.nan, columns=overlap, index=required_new
    )

    log_timing("inner", "update_summary_heatmap-nae_transpose")

    # ####### Create new vs new NumPy arrays, fill with calculated data. Then convert
    # to DFs, and append those to existing vs new DF, to create all vs new DFs #######

    new_against_new_corr = np.full(
        shape=[len(required_new), len(required_new)], fill_value=np.nan
    )
    new_against_new_pval = np.full(
        shape=[len(required_new), len(required_new)], fill_value=np.nan
    )
    new_against_new_logs = np.full(
        shape=[len(required_new), len(required_new)], fill_value=np.nan
    )

    log_timing("inner", "update_summary_heatmap-nan_init")

    for (v2_idx, v2) in enumerate(required_new):
        for (v1_idx, v1) in enumerate(required_new):
            if np.isnan(new_against_new_corr[v1_idx, v2_idx]):
                # Mask out any pairs that contain nans (this is done pairwise rather
                # than using .dropna on the full dataframe)
                mask = ~np.isnan(np_req[:, v1_idx]) & ~np.isnan(np_req[:, v2_idx])
                c, p = stats.pearsonr(np_req[mask, v1_idx], np_req[mask, v2_idx])
                new_against_new_corr[v1_idx, v2_idx] = c
                new_against_new_corr[v2_idx, v1_idx] = c
                new_against_new_pval[v1_idx, v2_idx] = p
                new_against_new_pval[v2_idx, v1_idx] = p
                if v1 != v2:
                    if required_new.index(v1) < required_new.index(v2):
                        new_against_new_logs[v1_idx, v2_idx] = -np.log10(p)
                    else:
                        new_against_new_logs[v2_idx, v1_idx] = -np.log10(p)

    log_timing("inner", "update_summary_heatmap-nan_calc")

    existing_against_new_corr[required_new] = pd.DataFrame(
        data=new_against_new_corr, columns=required_new, index=required_new
    )
    existing_against_new_pval[required_new] = pd.DataFrame(
        data=new_against_new_pval, columns=required_new, index=required_new
    )
    existing_against_new_logs[required_new] = pd.DataFrame(
        data=new_against_new_logs, columns=required_new, index=required_new
    )

    log_timing("inner", "update_summary_heatmap-nan_copy")

    ########
    # Append all vs new DFs to all vs existing DFs to give all vs all DFs.
    ########

    corr = corr.append(existing_against_new_corr)
    pvalues = pvalues.append(existing_against_new_pval)
    logs = logs.append(existing_against_new_logs)

    log_timing("inner", "update_summary_heatmap-ean_append", restart=False)
    log_timing("recalculate_corr_etc", "update_summary_heatmap-corr", restart=False)

    return corr, pvalues, logs


@app.callback(
    [
        Output("heatmap", "figure"),
        Output("corr-loaded-div", "children"),
        Output("pval-loaded-div", "children"),
    ],
    [Input("heatmap-dropdown", "value"), Input("heatmap-clustering-input", "value")],
    [State("df-loaded-div", "children")],
    prevent_initial_call=True,
)
@timing
def update_summary_heatmap(dropdown_values, clusters, df_loaded):
    logging.info(f"update_summary_heatmap {dropdown_values} {clusters}")
    # Guard against the first argument being an empty list, as happens at first
    # invocation, or df_loaded being False
    if df_loaded is False or len(dropdown_values) <= 1:
        fig = go.Figure()
        return fig, False, False

    # Load main dataframe
    dff = load("filtered")

    # Guard against the dataframe being empty
    if dff.size == 0:
        fig = go.Figure()
        return fig, False, False

    # Load data from previous calculation
    corr_dff = load("corr")
    pval_dff = load("pval")
    logs_dff = load("logs")

    # Add the index back in as a column so we can see it in the table preview
    dff.insert(loc=0, column="SUBJECTKEY(INDEX)", value=dff.index)

    # The columns we want to have calculated
    selected_columns = list(dropdown_values)
    logging.debug(f"selected_columns {selected_columns}")

    corr, pvalues, logs = recalculate_corr_etc(
        selected_columns, dff, corr_dff, pval_dff, logs_dff
    )
    start_timer("update_summary_heatmap")

    corr.fillna(0, inplace=True)

    try:
        cluster = AgglomerativeClustering(
            n_clusters=min(clusters, len(selected_columns)),
            affinity="euclidean",
            linkage="ward",
        )
        cluster.fit_predict(corr)
        clx = cluster.labels_
    except ValueError:
        clx = [0] * len(selected_columns)

    log_timing("update_summary_heatmap", "update_summary_heatmap-cluster")

    # Save cluster number of each column to a DF and then to feather.
    cluster_df = pd.DataFrame(data=clx, index=corr.index, columns=["column_names"])
    logging.debug(f"{cluster_df}")
    store("cluster", cluster_df)

    # TODO: what would be good here would be to rename the clusters based on the
    #  average variance (diags) within each cluster - that would reduce the
    #  undesirable behaviour whereby currently the clusters can jump about when
    #  re-calculating the clustering. Sort DFs' columns/rows into order based on
    #  clustering
    sorted_column_order = [x for _, x in sorted(zip(clx, corr.index))]

    sorted_corr = reorder_df(corr, sorted_column_order)
    sorted_corr = sorted_corr[sorted_corr.columns].apply(pd.to_numeric, errors="coerce")

    sorted_pval = reorder_df(pvalues, sorted_column_order)
    sorted_pval = sorted_pval[sorted_pval.columns].apply(pd.to_numeric, errors="coerce")

    sorted_logs = reorder_df(logs, sorted_column_order)
    sorted_logs = sorted_logs[sorted_logs.columns].apply(pd.to_numeric, errors="coerce")

    log_timing("update_summary_heatmap", "update_summary_heatmap-reorder")

    # Send to feather files
    store("corr", sorted_corr)
    store("pval", sorted_pval)
    store("logs", sorted_logs)

    flattened_logs = flattened(logs)
    store("flattened_logs", flattened_logs)

    log_timing("update_summary_heatmap", "update_summary_heatmap-save")

    # Remove the upper triangle and diagonal
    triangular = sorted_corr.to_numpy()
    triangular[np.tril_indices(triangular.shape[0], 0)] = np.nan
    triangular_pval = sorted_pval.to_numpy()
    triangular_pval[np.tril_indices(triangular_pval.shape[0], 0)] = np.nan

    log_timing(
        "update_summary_heatmap",
        "update_summary_heatmap-triangular",
        restart=False,
    )

    fig = go.Figure(
        go.Heatmap(
            z=np.fliplr(triangular),
            x=sorted_corr.columns[-1::-1],
            y=sorted_corr.columns[:-1],
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            customdata=np.fliplr(triangular_pval),
            hovertemplate=(
                "%{x}<br>"
                "vs.<br>"
                "%{y}<br>"
                "      r: %{z:.2g}<br>"
                " pval: %{customdata:.2g}<extra></extra>"
            ),
            colorbar_title_text="r",
            hoverongaps=False,
        ),
    )

    fig.update_layout(
        xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor="rgba(0,0,0,0)"
    )

    # Find the indices where the sorted classes from the clustering change.
    # Use these indices to plot vertical lines on the heatmap to demarcate the different
    # categories visually
    category_edges = np.concatenate((np.array([0]), np.diff(sorted(clx))))
    fig.update_layout(
        shapes=[
            dict(
                type="line",
                yref="y",
                y0=-0.5,
                y1=len(sorted_corr.columns) - 1.5,
                xref="x",
                x0=len(sorted_corr.columns) - float(i) - 0.5,
                x1=len(sorted_corr.columns) - float(i) - 0.5,
            )
            for i in np.where(category_edges)[0]
        ]
    )

    print_timings()

    return fig, True, True
