import logging

from visdex.app import app

# Default logging to stdout
logging.basicConfig(level=logging.INFO)
app.run_server(debug=False)
