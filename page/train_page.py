
def prediction():
	
	import streamlit as st
	import pandas as pd
	import torch
	from cadlae.detector import AnomalyDetector
	from cadlae.preprocess import DataProcessor
	from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, classification_report
	import numpy as np
	
	import matplotlib.pyplot as plt
	import seaborn as sns
	import dill
	import numpy as np
	import time
	from page.long_form_text import training_text
	
	training_text()
	
	st.sidebar.header('Set Model Parameters 🧪')
	batch_size = st.sidebar.slider('Select the batch size', 32, 512, 256, 32)
	epochs = st.sidebar.slider('Select the number of epochs', 5, 25, 10, 1)
	# select box for learning rate
	learning_rate = st.sidebar.selectbox('Select the learning rate', [0.001, 0.00001, 0.0001, 0.01])
	hidden_size = st.sidebar.slider('Select the hidden size', 10, 35, 25, 5)
	num_layers = st.sidebar.slider('Select the number of layers', 1, 3, 1, 1)
	sequence_length = st.sidebar.slider('Select the sequence length', 10, 50, 20, 5)
	dropout = st.sidebar.slider('Select the dropout', 0.1, 0.5, 0.2, 0.1)
	# true false
	use_bias = st.sidebar.checkbox('Use bias', value=True)
	
	if st.button('Train the Model! 🚀'):
		
		
	
		
		model = AnomalyDetector(batch_size=batch_size, num_epochs=epochs, lr=learning_rate,
								hidden_size=hidden_size, n_layers=num_layers, dropout=dropout,
								sequence_length=sequence_length, use_bias=use_bias,
								train_gaussian_percentage=0.25)
		
		train_link = "data/train_data.csv"
		test_link = "data/test_data_idv4.csv"
		processor = DataProcessor(train_link, test_link, "Fault", "Unnamed: 0")
		X_train = processor.X_train
		y_train = processor.y_train
		X_test = processor.X_test
		y_test = processor.y_test
		scaler = processor.scaler_function
		with st.spinner('Model is Training, Please Wait...'):
			model.fit(X_train)
			y_pred, details = model.predict(X_test, y_test)
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
		accuracy = round(accuracy * 100, 2)
		precision = round(precision * 100, 2)
		recall = round(recall * 100, 2)
		f1 = round(f1 * 100, 2)
		roc = round(roc * 100, 2)
		
		accuracy = str(accuracy) + '%'
		precision = str(precision) + '%'
		recall = str(recall) + '%'
		f1 = str(f1) + '%'
		roc = str(roc) + '%'
		
		# add metrics to dataframe, with columns Metric and Value
		metrics = pd.DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score'],
								'Result': [accuracy, precision, recall, f1, roc]})
		st.table(metrics)
		
		st.subheader('Classification Report 📝')
		report = classification_report(y_test, y_pred, output_dict=True)
		report = pd.DataFrame(report).transpose()
		st.table(report)
		
		st.subheader('Confusion Matrix')
		
		# plot confusion matrix
		def make_confusion_matrix(y_true, y_prediction, c_map="viridis"):
			sns.set(font_scale=0.8)
			fig, ax = plt.subplots()
			ax.set_title('CADLAE Confusion Matrix')
			cm = confusion_matrix(y_true, y_prediction)
			
			cm_matrix = pd.DataFrame(data=cm, columns=['Normal', 'Attack'],
									 index=['Normal', 'Attack'])
			
			sns.heatmap(cm_matrix, annot=True, fmt='.0f', cmap=c_map, linewidths=1, linecolor='black', clip_on=False)
			st.pyplot(fig)
		
		make_confusion_matrix(y_test, y_pred, c_map="Blues")
		
		st.subheader('ROC-AUC Curve')
		
		def plot_roc_curve(fpr, tpr):
			fig, ax = plt.subplots()
			ax.set_facecolor('white')  # set background color to white
			ax.spines['bottom'].set_color('black')  # set color of x-axis to black
			ax.spines['left'].set_color('black')
			plt.plot(fpr, tpr, color='orange', label='ROC')
			plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('Receiver Operating Characteristic (ROC) Curve')
			st.pyplot(fig)
		
		fpr, tpr, thresholds = roc_curve(y_test, y_pred)
		plot_roc_curve(fpr, tpr)
		
		def get_attacks(y_test, outlier=1, normal=0, breaks=[]):
			'''
			Get indices of anomalies
			:param y_test: predictions from semi supervised model
			:param outlier: label for anomalies
			:param normal: label for normal data points
			:param breaks: indices of breaks in data
			:return:
			'''
			events = dict()
			label_prev = normal
			event = 0  # corresponds to no event
			event_start = 0
			for tim, label in enumerate(y_test):
				if label == outlier:
					if label_prev == normal:
						event += 1
						event_start = tim
					elif tim in breaks:
						# A break point was hit, end current event and start new one
						event_end = tim - 1
						events[event] = (event_start, event_end)
						event += 1
						event_start = tim
				
				else:
					# event_by_time_true[tim] = 0
					if label_prev == outlier:
						event_end = tim - 1
						events[event] = (event_start, event_end)
				label_prev = label
			
			if label_prev == outlier:
				event_end = tim - 1
				events[event] = (event_start, event_end)
			return events
		
		def get_attack_idx_list(dictionary):
			'''
			Get list of indices of anomalies
			:param dictionary: dictionary of anomalies
			:return: Dictionary of anomalies, value is changed from (start, end) to list of indices
			'''
			for key, value in dictionary.items():
				if isinstance(value, tuple):
					dictionary[key] = list(range(value[0], value[1] + 1))
			return dictionary
		
		dict_attacks = get_attacks(y_pred, outlier=1, normal=0, breaks=[])
		attacks = get_attack_idx_list(dict_attacks)
		
		import matplotlib.pyplot as plt
		st.subheader('Predicted Anomalies')
		
		def plot_anomalies(df, column, anomalies, scaler=None):
			'''
			Plot anomalies
			:param df: dataframe to plot
			:param column: column to plot
			:param anomalies: dictionary of anomalies -> pass through dictionary - list pipeline to get dictionary with indx of anomalies
			:param reverse_scaler: object used to scale the data -> reverses to original scale in plot of passed
			'''
			fig, ax = plt.subplots()
			ax.set_facecolor('white')  # set background color to white
			ax.spines['bottom'].set_color('black')  # set color of x-axis to black
			ax.spines['left'].set_color('black')
			if scaler is not None:
				df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)
			title = "Plot of {}".format(column)
			ax.plot(df[column])
			ax.set_title(title)
			for key, value in anomalies.items():
				ax.plot(value, df[column][value], 'ro', markersize=4, color='red')
			st.pyplot(fig)
		
		plot_anomalies(X_test, "XMV(10)", attacks, scaler)
		
		st.subheader('LSTM Reconstruction Error')
		
		def plot_reconstructions(details, X, column):
			fig, ax = plt.subplots()
			ax.set_facecolor('white')  # set background color to white
			ax.spines['bottom'].set_color('black')  # set color of x-axis to black
			ax.spines['left'].set_color('black')
			ax.plot(X[column], label='original series')
			col_idx = X.columns.get_loc(column)
			ax.plot(details['errors_mean'][col_idx], label='reconstructed error mean', color='red')
			ax.set_title('Reconstructions of column {}'.format(column))
			st.pyplot(fig)
		
		plot_reconstructions(details, X_test, "XMV(10)")
