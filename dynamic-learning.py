# Dependencies
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import time
import sys
import datetime

# Loading Plant and drop unuseful row (with zero value or nan)	
df = pd.DataFrame()
dataset = pd.DataFrame()
data = pd.read_pickle('System_Plant3')
data = data.dropna()
cnt = 0 
dataset_size = 0 														#size of dataset
start_day_skipper = 0 													#indicate the days we need to skip
dataset_size_limit = 500 												#indicate limitation size of dataset for learning
saving_add = "/home/aryan/Desktop/plots-cooling/"

# Learning threat which is learning the Dataset we created before 
def learning(dataset,date_full):

	# Preprocessing for time show in plot
	date_full = date_full - datetime.timedelta(hours = (date_full.hour - 6))
	if date_full.minute > 30:
		date_full = date_full - datetime.timedelta(minutes = 30)
	S = []
	R = []
	E = []
	P = []
	
	# Sorting dataframe(dataset) in order to our usage(2 first column for input, 2 last column for output)
	dataset = dataset[['T0','Delta_t','T1','Delta_T']]
	X_size = 2
	values = list(dataset.columns.values)
	X = dataset[values[0:2]]
	X = np.array(X, dtype='float32')
	Y_size = 1
	values = list(dataset.columns.values)
	Y = dataset[values[3:4]]
	Y = np.array(Y, dtype='float32')
	
	# Shuffling  data to imoprove performance of learning algorithm
	indices = np.random.choice(len(X), len(X), replace=False)
	X_values = X[indices]
	Y_values = Y[indices]
	
	# Creating a train and a test dataset
	test_size = (dataset.size/4)/5
	X_test = X_values[-test_size:]
	X_train = X_values[:-test_size]
	Y_test = Y_values[-test_size:]
	Y_train = Y_values[:-test_size]
	
	# Creating a session of Tensor initialize the parameters
	sess = tf.Session()
	interval = 1
	epoch = 5000
	learning_rate = 0.01

	# Initialize placeholders size of hidden layers
	X_data = tf.placeholder(shape=[None, X_size], dtype=tf.float32)
	Y_target = tf.placeholder(shape=[None, Y_size], dtype=tf.float32)
	hidden_layer_nodes = ((dataset.size/4)/400)+1
	hidden_layer_nodes1 = ((dataset.size/4)/400)+1
	hidden_layer_nodes2 = ((dataset.size/4)/400)+1
	hidden_layer_nodes3 = ((dataset.size/4)/400)+1
	hidden_layer_nodes4 = ((dataset.size/4)/400)+1
		
	# Create variables for Neural Network layers(we using 4 hidden layer as we tested)
	w1 = tf.Variable(tf.random_normal(shape=[X_size,hidden_layer_nodes])) 					# Inputs -> Hidden Layer
	b1 = tf.Variable(tf.zeros(shape=[hidden_layer_nodes]))   								# First Bias
	w2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,hidden_layer_nodes1])) 		# Hidden layer -> Hidden layer1
	b2 = tf.Variable(tf.zeros(shape=[hidden_layer_nodes1]))   								# Second Bias
	w3 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes1,hidden_layer_nodes2])) 	# Hidden layer1 -> Hidden layer2
	b3 = tf.Variable(tf.zeros(shape=[hidden_layer_nodes2]))   								# Third Bias
	w4 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes2,hidden_layer_nodes3])) 	# Hidden layer2 -> Hidden layer3
	b4 = tf.Variable(tf.zeros(shape=[hidden_layer_nodes3]))   								# Fourth Bias
	w5 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes3,hidden_layer_nodes4])) 	# Hidden layer3 -> Hidden layer4
	b5 = tf.Variable(tf.zeros(shape=[hidden_layer_nodes4]))   								# Fourth Bias
	w6 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes4,Y_size])) 					# Hidden layer4 -> Outputs
	b6 = tf.Variable(tf.zeros(shape=[Y_size]))   											# Last Bias

	# Operations of hidden layer(all activation funstions are Sigmoid except last one which is linear)
	hidden_output = tf.sigmoid(tf.add(tf.matmul(X_data, w1), b1))
	hidden_output1 = tf.sigmoid(tf.add(tf.matmul(hidden_output, w2), b2))
	hidden_output2 = tf.sigmoid(tf.add(tf.matmul(hidden_output1, w3), b3))
	hidden_output3 = tf.sigmoid(tf.add(tf.matmul(hidden_output2, w4), b4))
	hidden_output4 = tf.sigmoid(tf.add(tf.matmul(hidden_output3, w5), b5))
	final_output = tf.add(tf.matmul(hidden_output4, w6),b6)
	
	# Cost function is MSE, Optimizer is Adam optimizer as we tested is better than Gradiant descent
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(learning_rate, global_step,100, 0.96, staircase=True)
	loss = tf.reduce_mean(tf.square(final_output - Y_target))
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step)
	
	# Initializing the Tensor variable
	init = tf.global_variables_initializer()
	sess.run(init)
	print "Dataset size =", dataset.size/4	
		
	# Training the model as we setted up
	print 'Training the model...'
	i = 0
	lrn_cnt = 0
	lmt_cycle_learning = 3 												#indicate the number of cycle we force dataset to learn
	while 1:
		sess.run(optimizer, feed_dict={X_data: X_train, Y_target: Y_train})
		if i % interval == 0:
			x = sess.run(loss, feed_dict={X_data: X_train, Y_target: Y_train})
			print "Epoch =",i," | Loss =",x,"\r",
			E.append(float(x))
			P.append(float(i))
			y = x
		if i == epoch :	
			print "Epoch =",i," | Loss =",x
			if x > 1:
				learning_rate = 0.1
				init = tf.global_variables_initializer()
				sess.run(init)
				i = 0
				lrn_cnt += 1
				if lrn_cnt > lmt_cycle_learning:
					print "Dataset is not proper for learning!"
					break
			else: break	
		if str(x) == "nan":
			print('Epoch', i, '|', 'Loss:', x)	
			print dataset
			init = tf.global_variables_initializer()
			sess.run(init)
		i += 1	
	# Plotting error 	
	plt.plot(P,E)
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.show(block=False)
	plt.pause(5)
	plt.close()	

	# Prediction future
	X_test[0][0] = dataset.iloc[0][0]
	t = 15 																#time step horizion
	X_test[0][1] = t
	cnt = 0
	i = 0
	last_pred = 0
	A = 0
	S.append(float(dataset.iloc[0][0]))
	R.append(str(date_full.strftime("%H:%M")))
	while i != 24:
		last_pred = sess.run(final_output, feed_dict={X_data: [X_test[0]]})
		S.append(float(last_pred + dataset.iloc[0][0]))
		R.append(str((date_full+datetime.timedelta(minutes=t)).strftime("%H:%M")))
		t = t + 15
		X_test[0][0] = dataset.iloc[0][0]
		#X_test[0][0] = last_pred
		X_test[0][1] = t 
		i += 1	
	print "Date =", str(date_full).split(" ")[0]
	plt.plot(R,S)
	plt.xlabel('Time (Minute)')
	plt.ylabel('Temperature (Celsius)')
	plt.xticks(rotation=90)
	plt.show(block=False)
	plt.savefig(saving_add + str(date_full).split(" ")[0] +".pdf", bbox_inches='tight')
	plt.savefig(saving_add + str(date_full).split(" ")[0] +".png", bbox_inches='tight')
	plt.close()
	tf.reset_default_graph()
	print "Graph saved."
	print "[Done!]","\n"
	
# Make Dataset from dataframe for learning the dynamic
for row in data.iterrows():
	if row[1][0] == 1: 													
		date_full = row[1][1]
		if dataset_size > dataset_size_limit:
			if cnt >= start_day_skipper:
				#print dataset
				learning(dataset,date_full)
			dataset.drop(dataset.index, inplace=True)
			df.drop(df.index, inplace=True)
			cnt += 1
			print cnt
		else:
			df.drop(df.index, inplace=True)
			dataset.drop(dataset.index, inplace=True)
			print "No data on ",date_full
	if row[1][1] != 0 and row[1][0] != 1:
		df = df.append(pd.DataFrame({'Delta_t':[row[1][0]],'T0':[row[1][1]],'T1':[row[1][2]]}), ignore_index=True)	
	else:
		if cnt >= start_day_skipper:
			for i in xrange(df.size/3):
				j = i
				k = 0
				while j != df.size/3:
					dataset = dataset.append(pd.DataFrame({'Delta_t':[(k+1)*5],'T0':[df.iloc[i][1]],'T1':[df.iloc[j][2]],'Delta_T':[df.iloc[j][2]-df.iloc[i][1]]}), ignore_index=True)	
					k = k+1
					j = j+1	
			dataset_size = dataset.size/4		
		else: 
			dataset_size = dataset_size_limit + 1
			pass




