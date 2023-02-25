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

def make_predictions():
    import streamlit as st
    import pandas as pd
    import torch
    from cadlae.detector import AnomalyDetector
    from cadlae.preprocess import DataProcessor
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time
    st.markdown(
        '''
        # Anomaly Detection 🤖
        Here we will train a model to detect anomalies in the Tennessee Eastman process, and then use the model to detect anomalies in the test set.
        '''
        
    )
    # button to make predictions
    st.markdown(
        """
		## Model Parameters Explained

		- `Batch Size` The number of samples to use in each batch
		- `Number of Epochs` The number of times to iterate over the entire dataset
		- `Learning Rate` The learning rate for the model
		- `Hidden Size` The number of nodes in the hidden layer
		- `Number of Layers` The number of layers in the model
		- `Dropout` The probability of randomly dropping out nodes during training to prevent overfitting
		- `Sequence Length` The number of time steps to use in each sequence
		- `Use Bias` Whether to include bias in the LSTM computations
	"""
    )
    

    st.sidebar.header('Set Model Parameters 🧪')
    batch_size = st.sidebar.slider('Select the batch size', 32, 512, 256, 32)
    epochs = st.sidebar.slider('Select the number of epochs', 5, 25, 10, 1)
    # select box for learning rate
    learning_rate = st.sidebar.selectbox('Select the learning rate', [0.00001, 0.0001,0.001 , 0.01])
    hidden_size = st.sidebar.slider('Select the hidden size', 10, 35, 25, 5)
    num_layers = st.sidebar.slider('Select the number of layers', 1, 3, 1, 1)
    sequence_length = st.sidebar.slider('Select the sequence length', 10, 50, 20, 5)
    dropout = st.sidebar.slider('Select the dropout', 0.1, 0.5, 0.2, 0.1)
    # true false
    use_bias = st.sidebar.checkbox('Use bias')

    
    if st.button('Train the Model! 🚀'):
        model = AnomalyDetector(batch_size=batch_size, num_epochs  = epochs, lr = learning_rate,
                                hidden_size = hidden_size, n_layers = num_layers, dropout = dropout,
                                sequence_length = sequence_length, use_bias = use_bias,
                                train_gaussian_percentage=0.25)
        train_link = "data/train_data.csv"
        test_link =  "data/test_data.csv"
        processor = DataProcessor(train_link, test_link, "Fault", "Unnamed: 0")
        X_train = processor.X_train
        y_train = processor.y_train
        X_test = processor.X_test
        y_test = processor.y_test
        scaler = processor.scaler_function
        with st.spinner('Model is Training, Please Wait...'):
            model.fit(X_train)
            y_pred, details = model.predict(X_test,y_test)
        st.header('Model Training Complete! 🎉')
        st.markdown('''
        The model has been trained in an unsupervised manner, and has learned the normal behaviour of the process.
        We have fit the model on unseen data using the parameters you have selected above. The results are shown below.
        ''')
        st.header('Model Performance 📈')
        st.subheader('Model Metrics 📊')
        cm = confusion_matrix(y_test, y_pred)
        TP = cm[0, 0]
        TN = cm[1, 1]
        FP = cm[0, 1]
        FN = cm[1, 0]
        accuracy = accuracy_score(y_test, y_pred)
        precision = TP / float(TP + FP)
        recall = TP / float(TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        roc = roc_auc_score(y_test, y_pred)
        
        # multiple *100, round to 2 decimal places, and add % sign
        accuracy = round(accuracy*100, 2)
        precision = round(precision*100, 2)
        recall = round(recall*100, 2)
        f1 = round(f1*100, 2)
        roc = round(roc*100, 2)
        
        accuracy = str(accuracy) + '%'
        precision = str(precision) + '%'
        recall = str(recall) + '%'
        f1 = str(f1) + '%'
        roc = str(roc) + '%'
        
        # add metrics to dataframe, with columns Metric and Value
        metrics = pd.DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score'],
                                'Result': [accuracy, precision, recall, f1, roc]})
        st.table(metrics)
        
        
        st.subheader('Confusion Matrix')
      

        # plot confusion matrix
        def make_confusion_matrix(y_true, y_prediction, c_map="viridis"):
            sns.set(font_scale=1.2)
            fig, ax = plt.subplots()
            ax.set_title('CADLAE Confusion Matrix')
            cm = confusion_matrix(y_true, y_prediction)
           
            cm_matrix = pd.DataFrame(data=cm, columns=['Normal', 'Attack'],
                                     index=['Normal', 'Attack'])
    
            sns.heatmap(cm_matrix, annot=True, fmt='.0f', cmap=c_map, linewidths=1, linecolor='black', clip_on=False)
            st.pyplot(fig)
        make_confusion_matrix(y_test,y_pred, c_map = "Blues")

      

        
        
        
            
        
        
    
    
        
        
    


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
    "Make Predictions": make_predictions,
    "DataFrame Demo": data_frame_demo
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()