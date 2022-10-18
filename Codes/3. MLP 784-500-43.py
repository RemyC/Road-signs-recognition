# ------------------------------------------- Create model -----------------------
#input layer
x0 = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 43])

#hidden layer
hidden_node=10
W1 = tf.Variable(tf.random_normal([784, hidden_node]))
b1 = tf.Variable(tf.random_normal([hidden_node]))
x1=tf.sigmoid(tf.matmul(x0, W1) + b1)

#output layer
W2 = tf.Variable (tf.random_normal ([784, 43])) # essai deuxieme couche
b2 = tf.Variable (tf.random_normal ([43]))
y = tf.nn.softmax (tf.matmul (x0, W2) + b2) # attention indice

# ------------------------------------------- Training ---------------------------
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
# initialise toutes les variables
init = tf.initialize_all_variables()
# definit la session
sess = tf.Session()
sess.run(init)

# -------------------------------------------- Evaluation ------------------------
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x0: testData, y_: testLabels}))

batch=100 #batchsize
epoch=200
for i in range(epoch):
    idx = np.arange(0, len(trainData)) # get all possible indexes
    np.random.shuffle(idx) # shuffle indexes
    idx = idx[0:batch] # use only `num` random indexes
    for j in idx:
        batch_xs = [trainData[j]] # get list of `num` random samples
        batch_xs = np.asarray(batch_xs) # get back numpy array
        batch_ys = [trainLabels[j]] # get list of `num` random samples
        batch_ys = np.asarray(batch_ys) # get back numpy array

    sess.run(train_step, feed_dict={x0: batch_xs, y_: batch_ys})
    print("[EPOCH "+str(i+1)+" /"+str(epoch)+")] {:.2f}%".format((sess.run(accuracy, feed_dict={x0: testData, y_: testLabels})) * 100))

prediction=np.empty(len(testLabels), dtype=np.int16)
prediction=(sess.run(tf.argmax(y,1), feed_dict={x0: testData, y_: testLabels}))
# --------------------------------------------------------------------------------