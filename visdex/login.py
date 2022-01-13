import logging

from flask_login import LoginManager, UserMixin, login_user, current_user

from dash import html, dcc
from dash.dependencies import Input, Output, State

from ldap3 import Server, Connection, ALL
import ldap

LDAP_SERVER="ldaps://uonauth.nottingham.ac.uk/"

class User(UserMixin):
    pass

USERS = {'bbzmsc': {}}

LOG = logging.getLogger(__name__)

def get_user(userid):
    if userid in USERS:
        user = User()
        user.id = userid
        return user
 
def check_password(userid, password):
    user = get_user(userid)    
    if user:
        try:
            ldap_client = ldap.initialize(LDAP_SERVER)
            # perform a synchronous bind
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
            ldap_client.simple_bind_s(dn, password)
            print("Authentication successful")
            return user
        except ldap.INVALID_CREDENTIALS:
            print('Authentication failed')
        except ldap.SERVER_DOWN:
            print('AD server not available')
        finally:
            ldap_client.unbind()

        # server = Server(LDAP_SERVER, use_ssl=True, get_info=ALL)
        # LOG.info("Server " + str(server))
        # ldap_username = 'cn=%s,ou=Staff,ou=Active,ou=Accounts,o=University' % userid
        # conn = Connection(server, ldap_username, password, auto_bind=False, raise_exceptions=False)
        # LOG.info("Conn" + str(conn))
        # try:
        #     LOG.info("Binding")
        #     conn.bind()
        #     LOG.info("Bound: %s" % conn.result)
        #     if conn.result['result'] == 0:  # Successful
        #         return user
        # except Exception as e:
        #     LOG.info("Exc ")
        #     LOG.warn(str(e))
        # except:
        #     LOG.info("???")
        # finally:
        #     LOG.info("Unbinding")
        #     if conn.bound:
        #         conn.unbind()

def init_login(flask_app):
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
            return '/visdex'

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
