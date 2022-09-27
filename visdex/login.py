"""
visdex: Login functionality
"""
import uuid

from flask import session
from flask_login import LoginManager, UserMixin, login_user

from dash import html, dcc
from dash.dependencies import Input, Output, State

import ldap

from visdex.common import Component

LDAP_SERVER="ldaps://uonauth.nottingham.ac.uk/"

class User(UserMixin):
    pass

class LoginPage(Component):
    def __init__(self, app):
        Component.__init__(self, app, id_prefix="data-", children=[
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

        self.register_cb(app, "try_login",
            Output('url_login', 'pathname'),
            [Input('login-button', 'n_clicks')],
            [State('uname-box', 'value'), State('pwd-box', 'value')]
        )

        self.register_cb(app, "login_outcome",
            Output('output-state', 'children'),
            [Input('login-button', 'n_clicks')],
            [State('uname-box', 'value'), State('pwd-box', 'value')]
        )

        self.known_users = app.server.config.get("KNOWN_USERS", [])
        self.log.info(f"Found %i known users", len(self.known_users))
        
        login_manager = LoginManager()
        login_manager.init_app(app.server)

        @login_manager.user_loader
        def user_loader(userid):
            return self._get_user(userid)
        
    def try_login(self, n_clicks, userid, password):
        user = self._check_password(userid, password)
        if user:
            login_user(user)
            session['uid'] = uuid.uuid4()
            self.log.info(f"Created session with UID {session['uid'].hex}")
            return f'{self.config.get("PREFIX", "/")}app'

    def login_outcome(self, n_clicks, userid, password):
        if n_clicks > 0:
            if self._check_password(userid, password):
                return ''
            else:
                return 'Incorrect userid or password'
        else:
            return ''

    def _get_user(self, userid):
        if userid in self.known_users:
            user = User()
            user.id = userid
            return user
    
    def _check_password(self, userid, password):
        user = self._get_user(userid)    
        if user:
            try:
                # Perform a synchronous bind to get Distinguished Name (DN)
                ldap_client = ldap.initialize(LDAP_SERVER)
                ldap_client.set_option(ldap.OPT_REFERRALS, 0)
                
                dns = ldap_client.search_st("ou=accounts,o=university", ldap.SCOPE_SUBTREE, filterstr='(uid=%s)' % userid, timeout=-1)
                if not dns:
                    self.log.warn("No matching user found: %s" % userid)
                    return

                dn = dns[0][0]
                if len(dns) > 1:
                    self.log.warn("Multiple matching DNs for user ID")
                    for dn, _attrs in dns:
                        self.log.warn(" - %s" % dn)
                else:
                    self.log.info("%s->%s" % (userid, dn))

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
