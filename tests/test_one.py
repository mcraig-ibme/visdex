import pytest

from dash.testing.application_runners import import_app
from selenium.webdriver.common.keys import Keys

# 2. give each testcase a tcid, and pass the fixture
# as a function argument, less boilerplate
@pytest.mark.skip(reason="Old test need to revise")
def test_basic_start(dash_duo):

    # 3. import the app inside the test function
    app = import_app('visdex.index')

    # 4. host the app locally in a thread, all dash server configs could be
    # passed after the first app argument
    dash_duo.start_server(app)

    # 5. use wait_for_* if your target element is the result of a callback,
    # keep in mind even the initial rendering can trigger callbacks
    dash_duo.wait_for_text_to_equal("#output-data-file-upload", "No file loaded", timeout=4)

    # 6. use this form if its present is expected at the action point
    assert dash_duo.find_element("#output-data-file-upload").text == "No file loaded"

    element = dash_duo.find_element('#missing-values-input')
    element.send_keys("1")
    assert dash_duo.find_element("#missing-values-input").get_attribute("value") == "1"
    element.send_keys(Keys.BACKSPACE)
    element.send_keys("2")
    assert dash_duo.find_element("#missing-values-input").get_attribute("value") == "2"

    # 7. to make the checkpoint more readable, you can describe the
    # acceptance criterion as an assert message after the comma.
    assert dash_duo.get_logs() == [], "browser console should contain no error"
