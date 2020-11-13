import itertools
import logging
import math
import numpy as np
import pandas as pd
import time
import scipy.stats as stats
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from psych_dashboard.app import app
from scipy.cluster.vq import kmeans, vq, whiten
from sklearn.cluster import AgglomerativeClustering
from psych_dashboard.load_feather import store, load
from psych_dashboard.timing import timing, timing_dict

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
    Convert a DF into a Series, where the MultiIndex of each element is a combination of the
    index/col from the original DF
    """
    # The series contains only half of the matrix, so filter by the order of the two level labels.
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
    ts = time.time()
    # Work out which columns/rows are needed anew, and which are already populated
    # TODO: note that if we load in a new file with some of the same column names, then this old
    #  correlation data may be used erroneously.
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

    te = time.time()
    timing_dict["update_summary_heatmap-init-corr"] = te - ts
    ts = time.time()
    # Populate missing elements in correlation matrix and p-values matrix using stats.pearsonr
    # Firstly, convert the dff columns needed to numpy (far faster than doing it each iteration)
    ts1 = time.time()

    np_dff_overlap = dff[overlap].to_numpy()
    np_dff_req = dff[required_new].to_numpy()

    te1 = time.time()
    timing_dict["update_summary_heatmap-numpy"] = te1 - ts1  # This is negligible

    ########
    # Create new vs existing NumPy arrays, fill with calculated data. Then convert to
    # DFs, and append those to existing vs existing DF, to create all vs existing DFs
    ########
    ts1 = time.time()
    new_against_existing_corr = np.full(
        shape=[len(overlap), len(required_new)], fill_value=np.nan
    )
    new_against_existing_pval = np.full(
        shape=[len(overlap), len(required_new)], fill_value=np.nan
    )
    new_against_existing_logs = np.full(
        shape=[len(overlap), len(required_new)], fill_value=np.nan
    )

    te1 = time.time()
    timing_dict["update_summary_heatmap-nae_init"] = te1 - ts1
    ts1 = time.time()

    for v2 in range(len(required_new)):
        for v1 in range(len(overlap)):
            # Mask out any pairs that contain nans (this is done pairwise rather than
            # using .dropna on the full dataframe)
            mask = ~np.isnan(np_dff_overlap[:, v1]) & ~np.isnan(np_dff_req[:, v2])
            c, p = stats.pearsonr(np_dff_overlap[mask, v1], np_dff_req[mask, v2])
            new_against_existing_corr[v1, v2] = c
            new_against_existing_pval[v1, v2] = p
            new_against_existing_logs[v1, v2] = -np.log10(p)

    te1 = time.time()
    timing_dict["update_summary_heatmap-nae_calc"] = te1 - ts1
    ts1 = time.time()

    new_against_existing_corr_df = pd.DataFrame(
        data=new_against_existing_corr, columns=required_new, index=overlap
    )
    corr[required_new] = new_against_existing_corr_df
    # logging.debug(f'corr {corr}')
    new_against_existing_pval_df = pd.DataFrame(
        data=new_against_existing_pval, columns=required_new, index=overlap
    )
    pvalues[required_new] = new_against_existing_pval_df
    # As new_against_existing_logs doesn't need to be transposed (the transpose is nans instead),
    # don't use an intermediate DF.
    logs[required_new] = pd.DataFrame(
        data=new_against_existing_logs, columns=required_new, index=overlap
    )

    te1 = time.time()
    timing_dict["update_summary_heatmap-nae_copy"] = te1 - ts1
    ts1 = time.time()

    ########
    # Create existing vs new DFs by transpose (apart from logs, whose transpose is nans)
    ########

    existing_against_new_corr = new_against_existing_corr_df.transpose()
    existing_against_new_pval = new_against_existing_pval_df.transpose()
    existing_against_new_logs = pd.DataFrame(
        data=np.nan, columns=overlap, index=required_new
    )

    te1 = time.time()
    timing_dict["update_summary_heatmap-nae_transpose"] = te1 - ts1
    ts1 = time.time()

    ########
    # Create new vs new NumPy arrays, fill with calculated data. Then convert to DFs, and append those to
    # existing vs new DF, to create all vs new DFs
    ########

    new_against_new_corr = np.full(
        shape=[len(required_new), len(required_new)], fill_value=np.nan
    )
    new_against_new_pval = np.full(
        shape=[len(required_new), len(required_new)], fill_value=np.nan
    )
    new_against_new_logs = np.full(
        shape=[len(required_new), len(required_new)], fill_value=np.nan
    )

    te1 = time.time()
    timing_dict["update_summary_heatmap-nan_init"] = te1 - ts1
    ts1 = time.time()

    for (v2_count, v2) in enumerate(required_new):
        for (v1_count, v1) in enumerate(required_new):
            if np.isnan(new_against_new_corr[v1_count, v2_count]):
                # Mask out any pairs that contain nans (this is done pairwise rather than using .dropna on the
                # full dataframe)
                mask = ~np.isnan(np_dff_req[:, v1_count]) & ~np.isnan(
                    np_dff_req[:, v2_count]
                )
                c, p = stats.pearsonr(
                    np_dff_req[mask, v1_count], np_dff_req[mask, v2_count]
                )
                new_against_new_corr[v1_count, v2_count] = c
                new_against_new_corr[v2_count, v1_count] = c
                new_against_new_pval[v1_count, v2_count] = p
                new_against_new_pval[v2_count, v1_count] = p
                if v1 != v2:
                    if required_new.index(v1) < required_new.index(v2):
                        new_against_new_logs[v1_count, v2_count] = -np.log10(p)
                    else:
                        new_against_new_logs[v2_count, v1_count] = -np.log10(p)

    te1 = time.time()
    timing_dict["update_summary_heatmap-nan_calc"] = te1 - ts1
    ts1 = time.time()

    existing_against_new_corr[required_new] = pd.DataFrame(
        data=new_against_new_corr, columns=required_new, index=required_new
    )
    existing_against_new_pval[required_new] = pd.DataFrame(
        data=new_against_new_pval, columns=required_new, index=required_new
    )
    existing_against_new_logs[required_new] = pd.DataFrame(
        data=new_against_new_logs, columns=required_new, index=required_new
    )

    te1 = time.time()
    timing_dict["update_summary_heatmap-nan_copy"] = te1 - ts1
    ts1 = time.time()

    ########
    # Append all vs new DFs to all vs existing DFs to give all vs all DFs.
    ########

    corr = corr.append(existing_against_new_corr)
    pvalues = pvalues.append(existing_against_new_pval)
    logs = logs.append(existing_against_new_logs)

    te1 = time.time()
    timing_dict["update_summary_heatmap-ean_append"] = te1 - ts1

    te = time.time()
    timing_dict["update_summary_heatmap-corr"] = te - ts

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
    # Guard against the second argument being an empty list, as happens at first invocation
    if df_loaded is True:
        dff = load("filtered")
        corr_dff = load("corr")
        pval_dff = load("pval")
        logs_dff = load("logs")

        # Add the index back in as a column so we can see it in the table preview
        if dff.size > 0 and len(dropdown_values) > 1:
            dff.insert(loc=0, column="SUBJECTKEY(INDEX)", value=dff.index)

            # The columns we want to have calculated
            selected_columns = list(dropdown_values)
            logging.debug(f"selected_columns {selected_columns}")

            corr, pvalues, logs = recalculate_corr_etc(
                selected_columns, dff, corr_dff, pval_dff, logs_dff
            )
            ts = time.time()

            corr.fillna(0, inplace=True)
            cluster_method = "hierarchical"
            if cluster_method == "Kmeans":
                # K-means clustering of correlation values.
                w_corr = whiten(corr)
                centroids, _ = kmeans(w_corr, min(7, int(math.sqrt(w_corr.shape[0]))))
                # Generate the indices for each column, which cluster they belong to.
                clx, _ = vq(w_corr, centroids)

            elif cluster_method == "hierarchical":
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
            else:
                raise ValueError

            te = time.time()
            timing_dict["update_summary_heatmap-cluster"] = te - ts
            ts = time.time()

            # Save cluster number of each column to a DF and then to feather.
            cluster_df = pd.DataFrame(
                data=clx, index=corr.index, columns=["column_names"]
            )
            logging.debug(f"{cluster_df}")
            store("cluster", cluster_df)

            # TODO: what would be good here would be to rename the clusters based on the average variance (diags) within
            # TODO: each cluster - that would reduce the undesirable behaviour whereby currently the clusters can jump about
            # TODO: when re-calculating the clustering.
            # Sort DFs' columns/rows into order based on clustering
            sorted_column_order = [x for _, x in sorted(zip(clx, corr.index))]

            sorted_corr = reorder_df(corr, sorted_column_order)
            sorted_corr = sorted_corr[sorted_corr.columns].apply(
                pd.to_numeric, errors="coerce"
            )

            sorted_pval = reorder_df(pvalues, sorted_column_order)
            sorted_pval = sorted_pval[sorted_pval.columns].apply(
                pd.to_numeric, errors="coerce"
            )

            sorted_logs = reorder_df(logs, sorted_column_order)
            sorted_logs = sorted_logs[sorted_logs.columns].apply(
                pd.to_numeric, errors="coerce"
            )

            te = time.time()
            timing_dict["update_summary_heatmap-reorder"] = te - ts
            ts = time.time()

            # Send to feather files
            store("corr", sorted_corr)
            store("pval", sorted_pval)
            store("logs", sorted_logs)

            flattened_logs = flattened(logs)
            store("flattened_logs", flattened_logs)

            te = time.time()
            timing_dict["update_summary_heatmap-save"] = te - ts
            ts = time.time()

            # Remove the upper triangle and diagonal
            triangular = sorted_corr.to_numpy()
            triangular[np.tril_indices(triangular.shape[0], 0)] = np.nan
            triangular_pval = sorted_pval.to_numpy()
            triangular_pval[np.tril_indices(triangular_pval.shape[0], 0)] = np.nan

            te = time.time()
            timing_dict["update_summary_heatmap-triangular"] = te - ts

            fig = go.Figure(
                go.Heatmap(
                    z=np.fliplr(triangular),
                    x=sorted_corr.columns[-1::-1],
                    y=sorted_corr.columns[:-1],
                    zmin=-1,
                    zmax=1,
                    colorscale="RdBu",
                    customdata=np.fliplr(triangular_pval),
                    hovertemplate="%{x}<br>vs.<br>%{y}<br>      r: %{z:.2g}<br> pval: %{customdata:.2g}<extra></extra>",
                    colorbar_title_text="r",
                    hoverongaps=False,
                ),
            )

            fig.update_layout(
                xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor="rgba(0,0,0,0)"
            )

            # Find the indices where the sorted classes from the clustering change. Use these indices to plot vertical
            # lines on the heatmap to demarcate the different categories visually
            y = np.concatenate((np.array([0]), np.diff(sorted(clx))))
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
                    for i in np.where(y)[0]
                ]
            )

            logging.info(f"{pd.DataFrame(timing_dict.items())}")

            return fig, True, True

    fig = go.Figure()

    logging.info(f"{pd.DataFrame(timing_dict.items())}")

    return fig, False, False
