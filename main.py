import streamlit as st
import pandas as pd
from PIL import Image
def intro():

    st.write("# Welcome to CADLAE 👋")
    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        CADLAE is a proposed framework for the anomaly detection, localization, and explanation of anomalies in Cyber Physical Systems:

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
		### Useful Links

		- Find the accompanying [Thesis](https://github.com/CameronLooney)
		- View the demo [Source Code](https://github.com/CameronLooney)
		- View the CADLAE [Source Code](https://github.com/CameronLooney)
		- Find me on [LinkedIn](https://www.linkedin.com/in/cameronlooney/)
	"""
    )

def mapping_demo():
    import streamlit as st
    import pandas as pd
    import pydeck as pdk

    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write(
        """
        This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/library/api-reference/charts/st.pydeck_chart)
to display geospatial data.
"""
    )


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

page_names_to_funcs = {
    "Home": intro,
    "Plotting Demo": plotting_demo,
    "Mapping Demo": mapping_demo,
    "DataFrame Demo": data_frame_demo
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()