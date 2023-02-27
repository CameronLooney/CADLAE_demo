def generate_corr_subgraph():
	import streamlit as st
	st.markdown(
		'''
		# Correlation Subgraph
		
		'''
	)
	batch_size = st.sidebar.slider('Select the minimum correlation', 0.1, 0.9, 0.6, 0.01)
	if st.button('Generate Correlation Subgraph! 🚀'):
		st.write("Minum correlation: ", batch_size)
		