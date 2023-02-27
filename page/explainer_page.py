def generation_explanation():
	import streamlit as st
	from cadlae.preprocess import DataProcessor
	from cadlae.explainer import ActionExplainer
	import matplotlib.pyplot as plt
	import streamlit.components.v1 as components
	import streamlit.components.v1 as components
	import streamlit as st
	import dtreeviz  # remember to load the package
	
	train_link = "data/train_data.csv"
	test_link = "data/test_data.csv"
	processor = DataProcessor(train_link, test_link, "Fault", "Unnamed: 0")
	X_train = processor.X_train
	y_train = processor.y_train
	X_test = processor.X_test
	y_test = processor.y_test
	col_names = processor.col_names
	st.markdown(
		'''
		# Anomaly Explanation 🚨
		
		### What is Anomaly Explanation?
		The proposed approach uses an unsupervised model to generate predictions, which are then used to train a
		supervised explanation model. The explanation model leverages Gradient Boosting Machines (GBMs) to
		generate interpretable and actionable rules in the form of if-then statements.
		GBMs are a popular machine learning model that iteratively adds decision trees to the model to
		improve prediction accuracy.

		### How does it work?
		The GBM model can be fitted to the training data by minimizing the loss function using gradient descent.
		GBMs can be interpreted by examining the importance of each feature in the model, which is calculated by
		measuring how much the model's accuracy decreases when a feature is randomly shuffled. GBMs also
		provide information about the contribution of each feature to each decision tree in the model.

		The generated if-then rules are expressed in terms of the input features and the predicted labels and
		can be transformed into actionable insights. The rules can be visualized easily due to the tree-based
		structure of the GBM, which allows the decision trees in the model to be displayed graphically.
		
		### What are the benefits?
		The importance of human-readable and actionable insights is that they enable workers in the CPS to quickly identify and correct anomalies, reducing the impact of the anomaly on the system.


		👈 **Pick the data point to describe and click the button to generate the explanation**


		'''
	)
	
	# streamlit text input type int between 0 and 10
	
	index = st.sidebar.number_input("Enter index of data point to explain", min_value=0, max_value=len(X_train), value=550, step=1)
	
	if st.button("Explain"):
		st.warning('Due to compatibility issues, the following charts are not dynamically sized, as a result, they may not be displayed correctly. We apologise for any inconvenience caused.')
		with st.spinner('Explainer is learning, Please Wait...'):
			mod = ActionExplainer()
			mod.fit(X_test, y_test, max_depth=4)
			mod.learn_data()
		
	
	
		def st_dtree(plot, height=None):
			#dtree_html = f"<body>{plot.view().svg()}</body>"
			dtree_html = f"<div style='text-align:center'><body>{plot.view().svg()}</body></div>"
			
			components.html(dtree_html, height=height, width=750)
		st.subheader('Global Explanation')
		st_dtree(mod.model, 550)
		
		def st_dtree_local(plot, height=None,width = None):
			dtree_html = f"<div style='text-align:center'><body>{plot.view(x=X_test.iloc[index], show_just_path=True).svg()}</body></div>"
			
			#dtree_html = f"<body>{plot.view(x=X_test.iloc[index], show_just_path=True).svg()}</body>"
			
			components.html(dtree_html, height=height, width=width)
		
		
		
		st.subheader('Local Explanation for data point {}'.format(index))
		st_dtree_local(mod.model, 600,800)
		prediction_proba = mod.clf.predict_proba(X_test.iloc[index].values.reshape(1, -1))
		if prediction_proba[0][0] > prediction_proba[0][1]:
			mode = "Normal Activity"
			prob = round(prediction_proba[0][0] * 100, 2)
		else:
			mode = "Fault"
			prob = round(prediction_proba[0][1] * 100, 2)
		
		st.markdown("""
		<style>
		.big-font {
		    font-size:18px !important;
		}
		</style>
		""", unsafe_allow_html=True)

		st.subheader('Report for data point: {} 🔍'.format(index))
	
		st.markdown('<p class="big-font">Model prediction for data point {}: <strong>{}</strong></p>'.format(index,mode), unsafe_allow_html=True)
		st.markdown('<p class="big-font">Probability of data point {} being in class {}: <strong>{}%.</strong></p>'.format(index, mode,prob),unsafe_allow_html=True)
		st.subheader('Suggested Action 💡')
		actions = mod.action(index)
		for action, description in actions.items():
			st.write('<p class="big-font"><strong>{}: </strong>{}</p>'.format(action,description), unsafe_allow_html=True)
		
