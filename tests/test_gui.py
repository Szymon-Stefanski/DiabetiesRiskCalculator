from streamlit.testing.v1 import AppTest


def test_app_runs():
    at = AppTest.from_file("../app/application.py").run()
    assert not at.exception


def test_app_title():
    at = AppTest.from_file("../app/application.py").run()
    assert at.title[0].value == "Diabetes Risk Assistant"


def test_prediction_flow():
    at = AppTest.from_file("../app/application.py").run()

    at.number_input[0].set_value(45)

    at.slider[0].set_value(13)

    at.button[0].click().run()

    assert not at.exception

    assert len(at.error) > 0 or len(at.success) > 0
