import os
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from psych_dashboard.app import app
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch, mm

PAGE_WIDTH, PAGE_HEIGHT = (210 * mm, 297 * mm)


def process_table(datatable):
    print(datatable)

    n_cols = len(datatable['props']['children']['props']['columns'])
    # Guard against 'format.specifier' not existing
    try:
        column_specifiers = [col['format']['specifier'] for col in
                             datatable['props']['children']['props']['columns']]
    except KeyError:
        column_specifiers = ['any'] * n_cols
    data_raw = datatable['props']['children']['props']['data']
    print(data_raw)

    # Process and append each row
    # Start with column headers
    column_names = [col['name'] for col in datatable['props']['children']['props']['columns']]
    table_contents = [column_names]
    for row in data_raw:
        row_contents = []
        # Handle each field in the row, applying the correct format specifier to match the DataTable.
        for val, spec in zip(list(row.values()), column_specifiers):
            try:
                row_contents.append(f'{val:{spec}}')
            except (TypeError, ValueError):
                if val is None:
                    row_contents.append('')
                else:
                    row_contents.append(f'{val}')
        table_contents.append(row_contents)

    print(table_contents)

    # Construct command sequence for TableStyle to align columns based upon data type
    # Guard against 'type' not existing
    try:
        column_types = [col['type'] for col in datatable['props']['children']['props']['columns']]
    except KeyError:
        column_types = ['any'] * n_cols

    command_sequence = []
    n_rows = len(data_raw)
    for col_number, col in enumerate(column_types):
        if col in ['numeric', 'datetime']:
            command_sequence.append(('ALIGN', (col_number, 1), (col_number, n_rows), 'RIGHT'))
        elif col in ['text', 'any']:
            command_sequence.append(('ALIGN', (col_number, 1), (col_number, n_rows), 'LEFT'))
        else:
            raise ValueError(col)

    column_widths = [100] + [(PAGE_WIDTH - 100) / n_cols] * (n_cols - 2) + [100]
    return Table(data=table_contents, style=TableStyle(command_sequence), colWidths=column_widths)


@app.callback(
    [Output('export-div', 'children')],
    [Input('export-pdf-button', 'n_clicks')],
    [State('heatmap', 'figure'),
     State('manhattan-figure', 'figure'),
     State('kde-figure', 'figure'),
     State('table_summary', 'children'),
     State('table_preview', 'children'),
     State({'type': 'gen-scatter-graph', 'index': ALL}, 'figure'),
     State({'type': 'gen-manhattan-graph', 'index': ALL}, 'figure'),
     ]
)
def export_to_pdf(n_clicks, *figs):
    if n_clicks is None:
        raise PreventUpdate
    output_directory = '/Users/samcox/Desktop'
    go.Figure(figs[0]).write_image(os.path.join(output_directory, 'test_heatmap.jpg'))
    go.Figure(figs[1]).write_image(os.path.join(output_directory, 'test_manhattan.jpg'))
    go.Figure(figs[2]).write_image(os.path.join(output_directory, 'test_kde.jpg'))
    tables = [process_table(fig) for fig in figs[3:5]]
    if len(figs[5]) > 0:
        for number, fig in enumerate(figs[5]):
            go.Figure(fig).write_image(os.path.join(output_directory, 'test_scatter'+str(number)+'.jpg'))
    if len(figs[6]) > 0:
        for number, fig in enumerate(figs[6]):
            go.Figure(fig).write_image(os.path.join(output_directory, 'test_man'+str(number)+'.jpg'))

    styles = getSampleStyleSheet()

    Title = "Dashboard Printout"
    pageinfo = ""

    def my_first_page(canvas, doc):
        canvas.saveState()
        canvas.setFont('Times-Bold', 16)
        canvas.drawCentredString(PAGE_WIDTH / 2.0, PAGE_HEIGHT - 108, Title)
        canvas.setFont('Times-Roman', 9)
        canvas.drawString(inch, 0.75 * inch, "First Page / %s" % pageinfo)
        canvas.restoreState()

    def my_later_pages(canvas, doc):
        canvas.saveState()
        canvas.setFont('Times-Roman', 9)
        canvas.drawString(inch, 0.75 * inch, "Page %d %s" % (doc.page, pageinfo))
        canvas.restoreState()

    def run_render():
        doc = SimpleDocTemplate(os.path.join(output_directory, "dashboard_printout.pdf"))
        Story = [Spacer(1, 2 * inch)]
        style = styles["Normal"]
        Story.append(Paragraph('Sample paragraph 1'))
        Story.append(Image(os.path.join(output_directory, 'test_heatmap.jpg'), kind='proportional', height=500, width=500))
        Story.append(Paragraph('Sample paragraph 2', style))
        Story.append(Image(os.path.join(output_directory, 'test_manhattan.jpg'), kind='proportional', height=500, width=500))
        Story.append(Paragraph('Sample paragraph 3', style))
        Story.append(Image(os.path.join(output_directory, 'test_kde.jpg'), kind='proportional', height=500, width=500))
        for table in tables:
            Story.append(table)
        if len(figs[5]) > 0:
            for number, fig in enumerate(figs[5]):
                Story.append(Paragraph(f'Scatter {number}'))
                Story.append(
                    Image(os.path.join(output_directory, 'test_scatter' + str(number) + '.jpg'), kind='proportional',
                          height=500, width=500))
        if len(figs[6]) > 0:
            for number, fig in enumerate(figs[6]):
                Story.append(Paragraph(f'Manhattan {number}'))
                Story.append(Image(os.path.join(output_directory, 'test_man' + str(number) + '.jpg'), kind='proportional', height=500, width=500))

        doc.build(Story, onFirstPage=my_first_page, onLaterPages=my_later_pages)

    run_render()

    return ['exported']
