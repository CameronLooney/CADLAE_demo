def generate_threshold_localisation():
	from cadlae.localisationFeatureWise import FeatureWiseLocalisation
	from cadlae.preprocess import DataProcessor
	from cadlae.detector import AnomalyDetector
	import streamlit as st
	
	train_link = "data/train_data.csv"
	test_link = "data/test_data.csv"
	processor = DataProcessor(train_link, test_link, "Fault", "Unnamed: 0")
	X_train = processor.X_train
	y_train = processor.y_train
	X_test = processor.X_test
	y_test = processor.y_test
	col_names = processor.col_names
	model = AnomalyDetector(batch_size=256, num_epochs=10, hidden_size=25)
	num_variables = st.sidebar.slider("Top K most likely variables", 1, len(col_names), 3)
	st.markdown(
		'''
		# Threshold Localisation 📌

		### What is Threshold Localisation?
		This section discusses how to use reconstruction error for each feature in time series data from an
		industrial control system to localize the cause of an anomaly. The process involves comparing the
		reconstruction error for each feature in the anomalous data to the maximum reconstruction error for that
		feature on the normal operation data.
		
		### How does it work?
		By setting a threshold value for each feature and identifying the features with the most time steps above
		the threshold, we can determine the features that are likely causing or closely related to the anomaly.


		👈 **Set the number of components you want returned and  click the button to generate the localisation**


		'''
	)

	
	
	if st.button("Feature Wise Localisation"):
		with st.spinner('Model is Training, Please Wait...'):
			model.fit(X_train)
			

		
		with st.spinner('Making Predictions, Please Wait...'):
			t_scores, d_train = model.predict(X_train)
			train_scores, details_train = t_scores.copy(), d_train.copy()
			test_preds, details_test = model.predict(X_test)
		
		with st.spinner("Using Predictions to Localise Anomalies..."):
			ftwise = FeatureWiseLocalisation(y_test, test_preds, processor.col_names, details_train, details_test)
			rank, y_predictions = ftwise.run()
		
			
		st.subheader("Top {} most likely causes of anomaly".format(num_variables))
		k = 3  # replace with the number of top features you want to print
		
		lst_sorted = sorted(rank, key=lambda x: x[1][0], reverse=True)[:k]  # sort by number of threshold violations
		for i, (feat, (violations, percentage)) in enumerate(lst_sorted):
			st.write(f"{i + 1}. {feat} with {violations} threshold violations ({percentage:.2f}%)")
	