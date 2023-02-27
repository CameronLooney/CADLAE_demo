def generate_threshold_localisation():
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