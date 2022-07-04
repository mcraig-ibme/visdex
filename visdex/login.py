"""
visdex: Login functionality
"""
import logging
import uuid

from flask import session
from flask_login import LoginManager, UserMixin, login_user, current_user

from dash import html, dcc
from dash.dependencies import Input, Output, State

import ldap

LDAP_SERVER="ldaps://uonauth.nottingham.ac.uk/"

class User(UserMixin):
    pass

KNOWN_USERS = []

LOG = logging.getLogger(__name__)

def get_user(userid):
    if userid in KNOWN_USERS:
        user = User()
        user.id = userid
        return user
 
def check_password(userid, password):
    user = get_user(userid)    
    if user:
        try:
            # Perform a synchronous bind to get Distinguished Name (DN)
            ldap_client = ldap.initialize(LDAP_SERVER)
            ldap_client.set_option(ldap.OPT_REFERRALS, 0)
            
            dns = ldap_client.search_st("ou=accounts,o=university", ldap.SCOPE_SUBTREE, filterstr='(uid=%s)' % userid, timeout=-1)
            if not dns:
                LOG.warn("No matching user found: %s" % userid)
                return

            dn = dns[0][0]
            if len(dns) > 1:
                LOG.warn("Multiple matching DNs for user ID")
                for dn, _attrs in dns:
                    LOG.warn(" - %s" % dn)
            else:
                LOG.info("%s->%s" % (userid, dn))

            # Now use DN to check credentials
            ldap_client.simple_bind_s(dn, password)
            print("Authentication successful")
            return user
        except ldap.INVALID_CREDENTIALS:
            print('Authentication failed')
        except ldap.SERVER_DOWN:
            print('AD server not available')
        finally:
            ldap_client.unbind()

def init(flask_app):
    global KNOWN_USERS
    KNOWN_USERS = flask_app.config.get("KNOWN_USERS", [])
    LOG.info(f"Found %i known users", len(KNOWN_USERS))
    
    login_manager = LoginManager()
    login_manager.init_app(flask_app)

    @login_manager.user_loader
    def user_loader(userid):
        return get_user(userid)

def get_layout(app):
    @app.callback(
        Output('url_login', 'pathname'),
        [Input('login-button', 'n_clicks')],
        [State('uname-box', 'value'), State('pwd-box', 'value')]
    )
    def _try_login(n_clicks, userid, password):
        user = check_password(userid, password)
        if user:
            login_user(user)
            session['uid'] = uuid.uuid4()
            LOG.info(f"Created session with UID {session['uid'].hex}")
            return f'{app.server.config.get("PREFIX", "/")}app'

    @app.callback(
        Output('output-state', 'children'),
        [Input('login-button', 'n_clicks')],
        [State('uname-box', 'value'), State('pwd-box', 'value')]
    )
    def _login_outcome(n_clicks, userid, password):
        if n_clicks > 0:
            if check_password(userid, password):
                return ''
            else:
                return 'Incorrect userid or password'
        else:
            return ''

    return html.Div([
        dcc.Location(id='url_login', refresh=True),
        html.H2("Please log in to continue:"),
        dcc.Input(placeholder='Enter your userid',
                  type='text',
                  id='uname-box'),
        dcc.Input(placeholder='Enter your password',
                  type='password',
                  id='pwd-box'),
        html.Button(children='Login',
                    n_clicks=0,
                    type='submit',
                    id='login-button'),
        html.Div(children='', id='output-state'),
    ])
