# Visdex - Visual Data Explorer

This is a Dash dashboard to explore tabluar data in a user-friendly manner.

## Installation

It is recommended to install the application in a virtual environment, for
example using Conda.

Visdex is a Dash application and can be run standalone for testing or with 
a WSGI compliant web application server. To run a standalone server the command
line application ```run-visdex``` is provided. However you should not use this
for externally accessible production use.

## Configuration

The main configuration file is located from the first of the following that
exists:

 - The VISDEX_CONFIG environment variable
 - $HOME/.visdex.conf
 - /etc/visdex/visdex.conf

The configuration file is a Python file and can use normal Python syntax for
dictionaries, lists, comments, etc.

### Configuration options

**SECRET_KEY**

This should be set to some unique string to enable the server to persist
sessions on restart

**DATA_CACHE_DIR**

This is the path to a directory where cached data files are stored. A directory
will be created here for each activate session. Data files in the session
cache directory will include filtered copies of the subset of data the user
is working on and possibly intermediate results of calculations for visualisation
plots.

**TIMEOUT_MINUTES**

Sessions will time out after being inactive for this period

**AUTH**

Dictionary describing the authentication/authorization system
in use. See ``config/visdex.example.conf`` for an example using
LDAP authentication. If not given, there will be no user access control.

**KNOWN_USERS**

List of usernames allowed to log in once authenticated. If not specified, no
control over who may log in. Note that data sources also can have their
own lists of users permitted to access each data source

**DATA_STORES**

Dictionary describing sources the data that may be visualised. 
Each key must be unique and map to a dictionary describing the data source.
As a minimum this should include a human-readable ``label``, and an
implementing ``class``. See ``config/visdex.example.conf`` for examples.

## Using the application

### Data selection

The top level dropdown menu gives a choice of data sources based on the list 
given in the configuration file (possibly restricted dependent on user ID).

Data sources may be of two broad types - server-side data which can be 
explored and queried by the user and user-supplied data which must be uploaded
to the server. 

#### Server-side data

Server side data is set up and configured as described in the 'Configuration'
section. Generally it will consist of one or more 'data sets' (tables) each
of which contains multiple fields/columns that the user may select for 
exploration.

Currently it is not possible to select fields from different tables, but this
is to be added soon.

#### User-supplied data

If user-supplied data is an option, the data should be in CSV
or TSV format - note that uploading of user data may or may not be enabled in
the configuration. If the application is available without authorization then
you may prefer not to allow users to upload their own data for security and
to avoid using too much server-side resources

A separate CSV/TSV file may also be uploaded containing a list of column names
to filter from the original data

## Summary tables and graphs

The summary section can be expanded and collapsed, and contains the following information

### Table Summary and Preview

The table summary shows a list of columns in the data with basic summary statistics
for each, including missing values.

Shows the first 5 rows of the data file (only the columns selected in the filter file
are shown) for information and to easily flag up some data read and formatting issues.

### Row/Column filter

The filtering section allows removal of columns based on proportion of missing values. 

It is also possible to filter rows on the value of the data the contain, for example
selecting only rows where a column has a value greater than some threshold.

Finally it is possible to generate a random sample of rows - useful for obtaining
a small data set for initial testing.

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

### Exploratory graph documentation

#### Scatter
Plot a Scatter graph to compare two variables. Options exist to encode further 
variables into colour and size of markers, and to split into separate graphs by
values of further variables.

Select a `regression degree` value to plot a regression line on each graph, 
with `r^2` displayed for each. 

#### Bar
Plot a Bar graph for a single variable, optionally split into value of a second 
variable.

Note that the Bar graph type is not suitable for float-type datasets. Instead, use the 
Histogram type in these instances. 

#### Manhattan
Plot a Manhattan graph for a single base variable against all other variables.
Select a reference p-value to see the correct p-value (taking into account the
number of correlations displayed) plotted as a horizontal line. Anything above
the horizontal line is potentially "significant".

#### Violin
Plot a Violin graph for a single variable. This displays a kernel density 
estimate, and box and whisker plot including mean.

#### Histogram
Plot a Histogram for a single variable.

- `n bins`: select the number of bins. Selecting `1` uses the default Plotly 
algorithm to determine the optimum number of bins.

### Adding/modifying exploratory graph types

Graph types can be modified by adding/removing/modifying input controls,
or by modifying the calculations performed in response.

New graph types can be also be added. This section discusses the code modifications 
required to do so. 

**NOTE THIS SECTION IS NOW OUT OF DATE AND NEEDS REVISING**

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
