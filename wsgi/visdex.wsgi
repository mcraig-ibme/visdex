import sys
import os
import logging

sys.path.insert(0,"/home/bbzmsc/.conda/envs/visdex/lib/python3.6/site-packages")
os.environ['SCRIPT_NAME'] = '/visdex/'

# Comment out to specify the configuration file if not in default location
#os.environ['VISDEX_CONFIG'] = '/etc/visdex/visdex.conf'

# Edit to define logging level
logging.basicConfig(filename="/home/bbzmsc/visdex_logs/visdex.log", level=logging.DEBUG)

from visdex.app import flask_app as application
