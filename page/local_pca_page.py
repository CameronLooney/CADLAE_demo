def generate_pca_localisation():
	from cadlae.localisationPCA import PCALocalization
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
	num_variables = st.sidebar.slider("Top K most likely variables", 1, 20, 5)
	st.markdown(
		'''
		# PCA Localisation 📌

		### What is PCA Localisation?
		Principal Component Analysis (PCA) is a statistical technique that transforms a dataset into a new
		coordinate system where the data is represented by a set of uncorrelated variables, known as principal
		components. The principal components are ordered in terms of their contribution to the variance in the data.
		
		### How does it work?
		PCA can be used for anomaly detection in cyber-physical systems by analyzing the reconstruction error
		from the test predictions of our proposed. The reconstruction error can be used to identify anomalous
		samples, and performing PCA on the reconstruction error allows us to identify the features that contributed
		the most to the anomaly. This information can be used to diagnose the cause of the anomaly and
		potentially take corrective action.

		👈 **Set the number of components you want returned and  click the button to generate the localisation**


		'''
	)
	if st.button("PCA Localisation"):
	
		
		
		with st.spinner('Model is Training, Please Wait...'):
			model.fit(X_train)
			
		with st.spinner('Making Predictions, Please Wait...'):
			y_pred, details = model.predict(X_test, y_test)
		
		with st.spinner("Using Predictions to Localise Anomalies..."):
			pca_localization = PCALocalization(3)
			pca_localization.fit(details["errors_mean"])
			#data_pca = pca_localization.transform(details["errors_mean"])
			result = pca_localization.localise(num_variables, col_names)
		
		st.subheader("Top {} most likely causes of anomaly".format(num_variables))
		for key,value in result.items():
			num = str(key) + ". "
			st.write(num,value)
			
		st.subheader("PCA Plot")
		with st.spinner("Plotting PCA..."):
			pca_localization = PCALocalization(3)
			pca_localization.pca_3D_st(details["errors_mean"], y_test)
		
		
		
			
		
			
	
			
			
			
			
			
			
			
		