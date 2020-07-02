import dash

app = dash.Dash(__name__, suppress_callback_exceptions=False)
server = app.server

indices = ['SUBJECTKEY', 'EVENTNAME']
