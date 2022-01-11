from flask_login import LoginManager, UserMixin, login_user, current_user

from dash import html, dcc
from dash.dependencies import Input, Output, State

class User(UserMixin):
    pass

USERS = {'bbzmsc': {'password': 'secret'}}

def get_user(userid):
    if userid in USERS:
        user = User()
        user.id = userid
        return user
 
def check_password(userid, password):
    user = get_user(userid)
    if user and password == USERS[userid]["password"]:
        return user

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
