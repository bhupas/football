
import pandas as pd
import numpy as np
import io
import sys
import os
from unittest.mock import MagicMock

# Mock streamlit
mock_st = MagicMock()
sys.modules['streamlit'] = mock_st
import streamlit as st

# Configure mocks
st.secrets = {"GEMINI_API_KEY": "fake_key"}
st.cache_data = lambda func: func
st.sidebar.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
st.columns.return_value = [MagicMock(), MagicMock()]

# We need a more dynamic side effect for columns since it's called with different args
def columns_side_effect(spec):
    if isinstance(spec, int):
        return [MagicMock() for _ in range(spec)]
    elif isinstance(spec, list):
        return [MagicMock() for _ in range(len(spec))]
    return [MagicMock(), MagicMock()]

st.sidebar.columns.side_effect = columns_side_effect
st.columns.side_effect = columns_side_effect
st.slider.return_value = 3 # Mock slider return value for k clusters

# Add current directory to path to import app
sys.path.append(os.getcwd())

# Import app after mocking
from app import load_and_prepare_data, perform_advanced_clustering

def test_load_data():
    file_path = 'SELVEVALUERING.xlsx'
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    with open(file_path, 'rb') as f:
        content = f.read()
    
    # Create a mock uploaded file object
    class MockUploadedFile:
        def __init__(self, name, content):
            self.name = name
            self.content = content
        
        def getvalue(self):
            return self.content

    uploaded_file = MockUploadedFile('SELVEVALUERING.xlsx', content)
    
    # Redirect stdout to a file
    with open('verification_output.txt', 'w') as log_file:
        sys.stdout = log_file
        
        print("Attempting to load data...")
        try:
            df = load_and_prepare_data([uploaded_file])
            if df is not None:
                print("Data loaded successfully!")
                
                # Now test the clustering function which generates the graph
                print("\nTesting perform_advanced_clustering...")
                
                # Mock plotly chart to inspect the figure
                mock_plotly_chart = MagicMock()
                st.plotly_chart = mock_plotly_chart
                
                perform_advanced_clustering(df)
                
                # Verify that plotly_chart was called
                if mock_plotly_chart.called:
                    print("st.plotly_chart was called.")
                    
                    # Inspect the figure passed to plotly_chart
                    # The first call should be the scatter plot
                    args, _ = mock_plotly_chart.call_args_list[0]
                    fig = args[0]
                    
                    print("Figure layout title:", fig.layout.title.text)
                    print("X-axis label:", fig.layout.xaxis.title.text)
                    print("Y-axis label:", fig.layout.yaxis.title.text)
                    
                    if "Offensiv Impact" in str(fig.layout.xaxis.title.text) and \
                       "Defensiv Impact" in str(fig.layout.yaxis.title.text):
                        print("SUCCESS: Graph axes updated correctly!")
                    else:
                        print("FAILURE: Graph axes do not match expected labels.")
                        
                else:
                    print("st.plotly_chart was NOT called.")
                        
            else:
                print("Failed to load data (returned None).")
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = sys.__stdout__

if __name__ == "__main__":
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")
    test_load_data()
