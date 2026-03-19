# Diabetes Risk Calculator

A student project that uses Machine Learning to estimate the risk of diabetes based on lifestyle and health indicators. 
The application was built using Python, Scikit-Learn, and Streamlit.

**Live Demo:** https://diabetes-risk-calculator.streamlit.app/

## Project Description
The application uses a classification model (Random Forest) trained on the CDC Diabetes Health Indicators dataset 
(over 70000 records). The goal of the project was to create a tool that visualizes a user's health profile and 
estimates the probability of disease risk in an accessible way.

### Key Features:
- Risk prediction (High/Low Risk) with percentage probability.
- Interactive radar chart presenting user data.
- Input data validation.
- Automated test suite for both the model and the interface.

## Tech Stack
- Python 3.11
- Scikit-Learn (Random Forest Model)
- Pandas, NumPy (Data Processing)
- Plotly, Streamlit (UI and Visualization)
- Pytest (Testing Framework)
- Docker (Containerization)

## Testing
The project includes unit and integration tests to ensure the correct operation of the algorithm and the UI.

To run the tests:
```bash
python -m pytest
