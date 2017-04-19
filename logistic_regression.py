#!/usr/bin/env python

import tensorflow as tf
import numpy as np

np.set_printoptions(threshold=np.nan)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, dimIN, dimOUT):

	W = tf.Variable(tf.random_normal((dimIN,dimOUT),stddev=0.01))
	b = tf.Variable(tf.random_normal((1, dimOUT),stddev=0.01))
	return tf.matmul(X, W) + b 


def HiddenLayer(X, dimIN, dimOUT):
    
    # ReLU hidden layer 
    w_h = tf.Variable(tf.random_normal([dimIN, dimOUT], stddev=0.01))
    b_h = tf.Variable(tf.random_normal([1, dimOUT], stddev=0.01))
    h = tf.nn.relu(tf.matmul(X, w_h)+b_h)

    return h


#Define Folder to run classifier on
folder = 'MSLR-WEB10K/Fold1/'

teX=np.load(folder+'testx.npy')
teY=np.load(folder+'testy.npy')
trX=np.load(folder+'trainx.npy')
trY=np.load(folder+'trainy.npy')
vaX=np.load(folder+'validx.npy')
vaY=np.load(folder+'validy.npy')

dimH1 = 100 #No. of units in hidden layer 1
dimH2 = 100 #No. of units in hidden layer 1
learning_rate = 0.0001 #Learning Rate of Optimizer 
epochs = 30 #Epochs to Run
batch_size = 128 
accuracy_old = 0

X = tf.placeholder("float", [None, 136]) 
Y = tf.placeholder("float", [None, 5])

#Uncomment either #1 or #2 to run classifier as simple logisitc classifier or neural network.

#1 Single Linear Layer Network
py_x = model(X, 136, 5)

#2 Hidden Layer Network
# h1 = HiddenLayer(X, 136, dimH1)
# h2 = HiddenLayer(h1, dimH1, dimH2)
# py_x = model(h2, dimH1, 5)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute mean cross entropy loss
#train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) # Stochastic Gradient Descent optimizer
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss) # Adam optimizer
predict_op = tf.argmax(py_x, 1) # predicted label

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(epochs):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)): #Mini Batch
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(trY, axis=1) ==sess.run(predict_op, feed_dict={X: trX})))    
        print(i, np.mean(np.argmax(vaY, axis=1) ==sess.run(predict_op, feed_dict={X: vaX})))
        print(i, np.mean(np.argmax(teY, axis=1) ==sess.run(predict_op, feed_dict={X: teX})))

        ltrue = np.argmax(teY, axis=1)
        #print (ltrue)
        lpred = sess.run(predict_op, feed_dict={X: teX})
        #print (lpred)

        accuracy = (np.mean(np.argmax(vaY, axis=1) ==sess.run(predict_op, feed_dict={X: vaX})))
        print accuracy

        unique,counts = np.unique(lpred,return_counts=True)
        #print counts

        if (accuracy >= accuracy_old):
            np.save(folder + str(learning_rate) + "truelabel.npy",ltrue) #Save True Relevance Labels
            np.save(folder + str(learning_rate) + "predlabel.npy",lpred)  #Save Predicted Relevance Labels
            print ("Model Updated") 
            accuracy_old = accuracy
    
    #ind = np.where(lpred == 4)	
	#print (ltrue[ind])
    print ("Best Validation Case:")
    print accuracy_old


