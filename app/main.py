import streamlit as st
import pandas as pd
import pickle as pkl


def main():
    st.set_page_config(
        page_title="Diabetes Prediction App", 
        page_icon=":diabetes:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.write("# Diabetes Prediction App")

if __name__ == "__main__":
    main()
