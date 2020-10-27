This is a Dash dashboard to explore data in a user-friendly manner.

# Installation

It is recommended to use Python 3.7 or higher.

Clone the repository to `$DASHBOARD_HOME`. 
It is _strongly recommended_ to use a virtual environment before installing.
```
python3 -m venv venv
source venv/bin/activate
pip install -e $DASHBOARD_HOME
```
On Windows: First make sure python 3 is installed and in your path
```
python -m venv venv
venv\Scripts\activate.bat
pip install -e $DASHBOARD_HOME
```
Run with
```
run-dashboard
```
This will begin a new app. In a browser, go to http://127.0.0.1:8050/ 
to view the app.

By default, the dashboard uses Feather format for data caching. If you
want to use Redis instead, you will need to set up a Redis server on port
6379, and set `use_redis = True` in `app.py`.

TODO: add redis install documentation?

# Documentation of functionality
The dashboard is split into 3 sections:
1. File selection
2. Summary tables and graphs
3. Exploratory graphs

## Data input
Upload Box1 should be used to select the main data file. This can be in a variety of 
formats (TODO add format specs).

Upload Box 2 can be used to select a (optional) filter file, which lists the columns to 
be used - TODO.

Selecting a file in either Upload Box will immediately parse the data in that file.

Click "Analyse" to run the analysis on the data.

## Summary tables and graphs
Select the `-` button to collapse the summary section.

### Table Preview
Shows the first 5 rows of the data file (only the columns selected in the filter file
are shown) for information and to easily flag up some data read and formatting issues.

### Table summary and filter
(?rename filter) Displays a summary of each 
column (min. max, quartiles, standard deviation etc), and uses colours to
highlight certain properties (TODO: expand that, document the colours in
the dashboard, and document them here). The filter box allows the user
to input the maximum percentage of rows in a column that are allowed to 
be missing (?or NA) before the column is removed from all the later 
analysis.

### Correlation heatmap
Displays the correlation matrix between all columns as a heatmap. Hover over 
items to view the Pearson correlation coefficient and p-value for that pair.
Use the dropdown to select/deselect columns to display, and use the
clusters input to select the number of clusters to separate the columns 
into. The clustering method used is Agglomerative Hierarchical Clustering:
https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering.
 
### Manhattan Plot
Displays a Manhattan plot of all variables.

Use the `pvalue` input to select the appropriate p-value threshold. The black 
horizontal line illustrates the associated `-log10` p-value to be used, 
corrected for the number of variables being compared. Anything above the line 
is considered 'significant' based upon the specified p-value threshold.

Toggle the `logscale y-axis` checkbox to toggle the y-axis scale.

A more customisable Manhattan plot against a single variable is available 
within the exploratory graphs. 

### Per-variable Histograms and KDEs
Show a histogram per variable. Overlaid onto each is a Gaussian Kernel 
Density Estimate of the variable. This only runs and updates when the 
'Run KDE analysis' checkbox is ticked - otherwise changes to the rest 
of the app make no change to the KDEs display.
(https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html).

## Exploratory graphs
"Exploratory graphs" is an area in which new, user-defined graphs can be 
created. Select "New Graph" to create a new graph. A Scatter ?area (TODO name?)
will be created, with a number of input controls and a blank graph area.
Use the input controls to create the graphs you wish to view.

Press `New Graph` repeatedly to create a new graph below the existing graph(s).
In this way, a collection of graphs can be built up.

Changing the `Graph type` on a graph will change it to another type, e.g.
Bar graph or Manhattan graph. Each graph type has a different collection
of input controls to curate the graph.

### Adding/modifying exploratory graph types
Graph types can be modified by adding/removing/modifying input controls,
or by modifying the calculations performed in response.

New graph types can be also be added. This section discusses the code modifications 
required to do so.

To add a new type of exploratory graphs (in this example, we add the imaginary graph 
type "Bubble"):

1. Add a new item to the `all_components` dictionary in `app.py` with the key being 
`bubble`. This item should be a list of dictionaries, one dictionary per input control. 
Each dictionary must as a minimum contain the keys 'component_type', 'id', and 'label'. 
'component_type' is a reference to the class of the component, e.g. `dcc.Dropdown`. 
'id' should be unique to that dictionary within the list. 'label' will be the 
text displayed to label the component on the dashboard. All other keys in the 
dictionary must be valid keyword arguments to the class constructor of the class in 
`component_type`, and will be passed as input keywords to the component.

2. Create `bubble_graph.py` in the `exploratory_graphs` directory, and populate it with 
`update_bubble_components()` and `make_bubble_figure()`, modelled upon those found in 
the other `*_graph.py` files. `update_bubble_components()` handles the recreation of 
each component whenever one is edited, most of which is handled by 
`update_graph_components()` in `exploratory_graph_groups.py`.

3. Import `bubble_graph` to `index.py`


# TODO:
- Where to put data
- Summary vs exploratory apps