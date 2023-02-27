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
		
			
		st.subheader("Top {} most likely causes of anomaly".format(10))
		st.write(rank)
	