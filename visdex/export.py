import os
import logging

from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import dash_html_components as html

import plotly.graph_objects as go

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch, mm

from visdex.common import all_components, div_style

PAGE_WIDTH, PAGE_HEIGHT = (210 * mm, 297 * mm)

LOG = logging.getLogger(__name__)

def get_layout(app):
    @app.callback(
        [Output("export-div", "children")],
        [Input("export-pdf-button", "n_clicks")],
        [
            State("heatmap", "figure"),
            State("manhattan-figure", "figure"),
            State("kde-figure", "figure"),
            State("table_summary", "children"),
            State("table_preview", "children"),
            *[
                State({"type": "gen-" + str(graph_type) + "-graph", "index": ALL}, "figure")
                for graph_type in all_components
            ],
        ],
    )
    def _export_to_pdf(n_clicks, *figs):

        if n_clicks is None:
            raise PreventUpdate

        output_directory = ""
        styles = getSampleStyleSheet()
        title = "Visual Data Explorer"
        pageinfo = ""

        tables = [process_table(fig) for fig in figs[3:5]]
        run_render(title, pageinfo, tables, figs, output_directory, styles)

        return ["exported"]

    return html.Div(children=[
        html.Button("Export to PDF", id="export-pdf-button", style=div_style),
        html.Div(children=[""], id="export-div", style=div_style),
    ])
 
def save_images(figs, output_directory):
    """
    Save each image to an intermediate file, which will then be included in the
    final report.
    """
    # Only save if the figure contains data
    for idx, fig in enumerate(figs):
        LOG.debug("%i: %s", idx, fig)

    if figs[0]["data"]:
        LOG.debug("write 0")
        go.Figure(figs[0]).write_image(
            os.path.join(output_directory, "summary_heatmap.jpg")
        )
    if figs[1]["data"]:
        LOG.debug("write 1")
        go.Figure(figs[1]).write_image(
            os.path.join(output_directory, "summary_manhattan.jpg")
        )
    if figs[2]["data"]:
        LOG.debug("write 2")
        go.Figure(figs[2]).write_image(
            os.path.join(output_directory, "summary_kde.jpg")
        )
    for list_of_graphs_of_type, graph_type in zip(
        figs[5:], list(all_components.keys())
    ):
        for number, fig in enumerate(list_of_graphs_of_type):
            LOG.debug("write %i", number)
            go.Figure(fig).write_image(
                os.path.join(
                    output_directory,
                    "exploratory_" + str(graph_type) + str(number) + ".jpg",
                )
            )

def run_render(title, pageinfo, tables, figs, output_directory, styles):
    def my_first_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Bold", 16)
        canvas.drawCentredString(PAGE_WIDTH / 2.0, PAGE_HEIGHT - 108, title)
        canvas.setFont("Times-Roman", 9)
        canvas.drawString(inch, 0.75 * inch, "First Page / %s" % pageinfo)
        canvas.restoreState()

    def my_later_pages(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 9)
        canvas.drawString(inch, 0.75 * inch, "Page %d %s" % (doc.page, pageinfo))
        canvas.restoreState()

    doc = SimpleDocTemplate(
        os.path.join(output_directory, "dashboard_printout.pdf")
    )
    Story = [Spacer(1, 2 * inch)]

    LOG.debug(figs)
    save_images(figs, output_directory)

    # Only include image if the figure contains data
    if figs[0]["data"]:
        Story.append(Paragraph("Correlation matrix:", style=styles["Heading1"]))
        Story.append(
            Image(
                os.path.join(output_directory, "summary_heatmap.jpg"),
                kind="proportional",
                height=500,
                width=500,
            )
        )
    if figs[1]["data"]:
        Story.append(
            Paragraph(
                "Manhattan graph with corrected p-value threshold",
                style=styles["Heading1"],
            )
        )
        Story.append(
            Image(
                os.path.join(output_directory, "summary_manhattan.jpg"),
                kind="proportional",
                height=500,
                width=500,
            )
        )
    if figs[2]["data"]:
        Story.append(Paragraph("KDE for each variable", style=styles["Heading1"]))
        Story.append(
            Image(
                os.path.join(output_directory, "summary_kde.jpg"),
                kind="proportional",
                height=500,
                width=500,
            )
        )

    for table in tables:
        Story.append(table)
    for list_of_graphs_of_type, graph_type in zip(
        figs[5:], list(all_components.keys())
    ):
        for number, _fig in enumerate(list_of_graphs_of_type):
            Story.append(
                Paragraph(f"{graph_type} {number}", style=styles["Heading2"])
            )
            Story.append(
                Image(
                    os.path.join(
                        output_directory,
                        "exploratory_" + str(graph_type) + str(number) + ".jpg",
                    ),
                    kind="proportional",
                    height=500,
                    width=500,
                )
            )

    doc.build(Story, onFirstPage=my_first_page, onLaterPages=my_later_pages)

def process_table(datatable):
    if datatable is None:
        return None

    n_cols = len(datatable["props"]["children"]["props"]["columns"])
    # Guard against 'format.specifier' not existing
    try:
        column_specifiers = [
            col["format"]["specifier"]
            for col in datatable["props"]["children"]["props"]["columns"]
        ]
    except KeyError:
        column_specifiers = ["any"] * n_cols
    data_raw = datatable["props"]["children"]["props"]["data"]

    # Process and append each row
    # Start with column headers
    column_names = [
        col["name"] for col in datatable["props"]["children"]["props"]["columns"]
    ]
    table_contents = [column_names]
    for row in data_raw:
        row_contents = []
        # Handle each field in the row, applying the correct format specifier to match
        # the DataTable.
        for val, spec in zip(list(row.values()), column_specifiers):
            try:
                row_contents.append(f"{val:{spec}}")
            except (TypeError, ValueError):
                if val is None:
                    row_contents.append("")
                else:
                    row_contents.append(f"{val}")
        table_contents.append(row_contents)

    # Construct command sequence for TableStyle to align columns based upon data type
    # Guard against 'type' not existing
    try:
        column_types = [
            col["type"] for col in datatable["props"]["children"]["props"]["columns"]
        ]
    except KeyError:
        column_types = ["any"] * n_cols

    command_sequence = []
    n_rows = len(data_raw)
    for col_number, col in enumerate(column_types):
        if col in ["numeric", "datetime"]:
            command_sequence.append(
                ("ALIGN", (col_number, 1), (col_number, n_rows), "RIGHT")
            )
        elif col in ["text", "any"]:
            command_sequence.append(
                ("ALIGN", (col_number, 1), (col_number, n_rows), "LEFT")
            )
        else:
            raise ValueError(col)

    column_widths = [100] + [(PAGE_WIDTH - 100) / n_cols] * (n_cols - 2) + [100]
    return Table(
        data=table_contents, style=TableStyle(command_sequence), colWidths=column_widths
    )


