import streamlit as st
import pandas as pd
from PIL import Image
import torch
from tqdm import trange
import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import logging
import pandas as pd
import torch.nn as nn
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_curve, roc_auc_score


from cadlae.detector import AnomalyDetector, DetectorHelper
model = AnomalyDetector()
def intro():
    

    st.write("# Welcome to CADLAE 👋")

    st.markdown(
        """
        Configurable Anomaly Detection, Localization, and Explanation Framework for Cyber Physical Systems

        **👈 Select a demo from the dropdown on the left to explore the framework**
        

        ### CADLAE Framework
        """)
    image = Image.open("images/architecture.png")
    # make image 0.75 times the size of the original
    image = image.resize((int(image.width * 0.75), int(image.height * 0.75)))
    st.image(image, use_column_width=True)

    st.markdown(
        '''
        ### The Tennessee Eastman Process
        '''
    )
    image = Image.open("images/TEP_diagram.png")
    # make image 0.75 times the size of the original
    image = image.resize((int(image.width * 0.75), int(image.height * 0.75)))
    # load image with boarder around it

    st.image(image, use_column_width=True)
    st.markdown(
        '''
        The Tennessee Eastman process has been widely used as a testbed to study various challenges faced in continuous
        processes. Originally proposed by Downs and Vogel (1993), the TEP has been used for plant-wide control design,
        multivariate control, optimisation, predictive control, adaptive control, nonlinear control, process diagnostics
        , and educational purposes.In recent years, many studies involving the TEP have focused on fault detection using
        classical statistics or machine learning methods.
        '''
    )
    st.markdown(
        """
		### Relevant Links

		- Find the accompanying [Thesis](https://github.com/CameronLooney)
		- View the demo [Source Code](https://github.com/CameronLooney)
		- View the CADLAE [Source Code](https://github.com/CameronLooney)
		- Find me on [LinkedIn](https://www.linkedin.com/in/cameronlooney/)
	"""
    )

def train_and_predict_page():
    from cadlae.detector import AnomalyDetector
    from page.train_page import prediction
    prediction()
    

        

      

        
        
        
            
        
        
    
    
        
        
    


def plotting_demo():
    import streamlit as st
    import time
    import numpy as np

    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.write(
        """
        This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!
"""
    )

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    chart = st.line_chart(last_rows)

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text("%i%% Complete" % i)
        chart.add_rows(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)

    progress_bar.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")

def correlation_subgraph():
    from page.corr_subgraph_page import generate_corr_subgraph
    generate_corr_subgraph()
def data_frame_demo():
    import streamlit as st
    import pandas as pd
    import altair as alt

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
    st.write(
        """
        This demo shows how to use `st.write` to visualize Pandas DataFrames.

(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)
"""
    )

    @st.cache_data
    def get_UN_data():
        AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
        df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
        return df.set_index("Region")

    try:
        df = get_UN_data()
        countries = st.multiselect(
            "Choose countries", list(df.index), ["China", "United States of America"]
        )
        if not countries:
            st.error("Please select at least one country.")
        else:
            data = df.loc[countries]
            data /= 1000000.0
            st.write("### Gross Agricultural Production ($B)", data.sort_index())

            data = data.T.reset_index()
            data = pd.melt(data, id_vars=["index"]).rename(
                columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
            )
            chart = (
                alt.Chart(data)
                .mark_area(opacity=0.3)
                .encode(
                    x="year:T",
                    y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                    color="Region:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**

            Connection error: %s
        """
            % e.reason
        )

def localise_pca():
    from page.local_pca_page import generate_pca_localisation
    generate_pca_localisation()
    
def localise_threshold():
    from page.local_threshold import generate_threshold_localisation
    generate_threshold_localisation()
    
def explainer():
    from page.explainer_page import generation_explanation
    generation_explanation()
    
def feedback_form():
    from page.feedback_form import feedback
    feedback()
    
def full_demo():
    from page.full_pipeline import pipeline
    pipeline()
page_names_to_funcs = {
    "Introduction": intro,
    "Full Demo": full_demo,
    "Train Model": train_and_predict_page,
    "Correlation Subgraph": correlation_subgraph,
    "PCA Localisation": localise_pca,
    "Threshold Localisation": localise_threshold,
    "Explanation" : explainer,
    "Feedback Form" : feedback_form,
    
   
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()