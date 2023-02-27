def generate_corr_subgraph():
	from cadlae.preprocess import DataProcessor
	from cadlae.correlationSubgraph import CorrelationSubgraph
	from cadlae.localisationSubgraph import LocaliseSubgraph
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.metrics import roc_curve,roc_auc_score
	import seaborn as sns
	import networkx as nx
	
	train_link = "data/train_data.csv"
	test_link = "data/test_data_idv4.csv"
	processor = DataProcessor(train_link, test_link, "Fault", "Unnamed: 0")
	X_train = processor.X_train
	import streamlit as st
	
	
		
	
	
	st.markdown(
		'''
		# Correlation Subgraph
		
		### What is a correlation subgraph?
		Here we describe how a correlation graph can be generated to localise the cause of an anomaly in a cyber physical system.
		To create this graph, we use the Spearman rank correlation coefficient, which is a non-parametric measure of correlation based
		on the ranks of the data. This coefficient is robust to non-normality and can handle both ordinal and continuous variables,
		making it suitable for use in a cyber physical system where data may not always be normally distributed.
		
		### How does it work?
		The correlation coefficient is calculated between all pairs of features in the system, and if the absolute value of their correlation coefficient
		is above a user-defined threshold, we create an edge between them in the correlation graph. This threshold is implemented
		to ensure disconnected subgraphs are generated. The resulting graph will consist of a set of disconnected
		subgraphs, where each subgraph is a group of features that are highly correlated with each other.
		These subgraphs can then be used to localise the cause of the anomaly.
		
		👈 **Set the minimum correlation threshold in the sidebar and click the button to generate the correlation subgraph.**


		'''
	)
	corr = st.sidebar.slider('Select the minimum correlation', 0.1, 0.9, 0.6, 0.01)
	if st.button('Generate Correlation Subgraph! 🚀'):
		st.subheader('Correlation Subgraph with threshold = ' + str(corr))
		subgraph = CorrelationSubgraph(X_train, corr)
		subgraph.plot_corr_graph_st()
		
		st.subheader('Subgraphs Generated')
		for key, value in subgraph.generate_subgraph_dict().items():
			st.write(str(key) + ': ' + ', '.join(value))
			
			
			
		st.subheader('Subgraph most likely to contain the cause of the anomaly {}'.format("d"))
			
		
		
		