import logging

from visdex.app import app

logging.getLogger().setLevel(logging.INFO)
app.run_server(debug=False)
