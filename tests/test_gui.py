from streamlit.testing.v1 import AppTest


def test_app_runs():
    """
    Smoke test to ensure the Streamlit application launches without any unhandled exceptions.
    It verifies that the basic execution flow of the script is correct.
    """
    at = AppTest.from_file("../app/application.py").run()
    assert not at.exception


def test_app_title():
    """
    Verifies that the application's main title is correctly rendered and matches
    the expected 'Diabetes Risk Assistant' string.
    """
    at = AppTest.from_file("../app/application.py").run()
    assert at.title[0].value == "Diabetes Risk Assistant"


def test_prediction_flow():
    """
    Tests the end-to-end prediction workflow of the application.
    Simulates user input for BMI and Age, triggers the analysis button,
    and verifies that a risk result (Success/Error message) is displayed
    without any runtime errors.
    """
    at = AppTest.from_file("../app/application.py").run()

    # Set BMI to a high value
    at.number_input[0].set_value(45)

    # Set Age Group to the highest category
    at.slider[0].set_value(13)

    # Trigger the 'Analyze Risk' button
    at.button[0].click().run()

    # Ensure no exceptions occurred during the prediction process
    assert not at.exception

    # Verify that the application displayed a prediction result
    assert len(at.error) > 0 or len(at.success) > 0
