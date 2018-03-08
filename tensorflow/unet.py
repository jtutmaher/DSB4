import tensorflow as tf
import numpy as np
import os

# MNIST DATA
batch_size=64
train_size=42000
length = 1024
intermediate = 1000
reg = 5e-4

# DEFINE SESSION
sess = tf.InteractiveSession()

# INPUTS
x = tf.placeholder("float",shape=[None,length])
y_ = tf.placeholder("float",shape=[None,length])
x_image = tf.reshape(x,[-1,32,32,1])
y_image = tf.reshape(y_,[-1,32,32,1])

def load_images(filename):
	return np.load(filename)['arr_0']

def load_labels(filename):
	return np.load(filename)['arr_0']

# DEFINE LAYERS
def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def bias_variable(shape):
	return tf.Variable(tf.constant(0.1,shape=shape))

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def deconv2d(x, W):
	x_shape = tf.shape(x)
	output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
	return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1], padding='SAME')

def output_conv2d(x,W):
	x_shape = tf.shape(x)
	output_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], 1])
	return tf.nn.conv2d_transpose(x,W,output_shape=output_shape,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def dice_coef(y_true, y_pred,smooth=1.):
	y_true_f = tf.layers.flatten(y_true)
	y_pred_f = tf.layers.flatten(y_pred)
	intersection = tf.reduce_sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# VARIABLES
W_conv1 = weight_variable([32,32,1,32])
b_conv1 = bias_variable([32])

# FIRST DOWN CONVOLUTIONAL LAYER
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# SECOND DOWN CONVOLUTIONAL LAYER
W_conv2 = weight_variable([16,16,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# THIRD DOWN CONVOLUTION LAYER
W_conv3 = weight_variable([8,8,64,128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# FOURTH DOWN CONVOLUTION LAYER
W_conv4 = weight_variable([4,4,128,256])
b_conv4 = bias_variable([256])

h_conv4 = tf.nn.relu(conv2d(h_pool3,W_conv4)+b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

# FIRST UP CONVOLUTION LAYER
W_conv5 = weight_variable([4,4,128,256])
b_conv5 = bias_variable([128])

h_conv5 = tf.nn.relu(deconv2d(h_conv4,W_conv5)+b_conv5)

# SECOND UP CONVOLUTION LAYER
W_conv6 = weight_variable([8,8,64,128])
b_conv6 = bias_variable([64])

h_conv6 = tf.nn.relu(deconv2d(h_conv5,W_conv6)+b_conv6)

# THIRD UP CONVOLUTION LAYER
W_conv7 = weight_variable([16,16,32,64])
b_conv7 = bias_variable([32])

h_conv7 = tf.nn.relu(deconv2d(h_conv6,W_conv7)+b_conv7)

# FOURTH UP CONVOLUTION
#W_conv8 = weight_variable([16,16,1,32])
#b_conv8 = bias_variable([1])

#h_conv8 = tf.nn.relu(deconv2d(h_conv7,W_conv8)+b_conv8)

# OUTPUT
W_out = weight_variable([32,32,1,32])
b_out = bias_variable([1])
output = tf.nn.sigmoid(output_conv2d(h_conv7,W_out)+b_out)

predict = output

# LOSS
loss = dice_coef_loss(predict,y_image)

# L2 REGULARIZATION
#regularizers = tf.nn.l2_loss(W_conv1)+tf.nn.l2_loss(W_conv2)+tf.nn.l2_loss(W_conv3)+tf.nn.l2_loss(W_conv4)+tf.nn.l2_loss(W_conv5)+tf.nn.l2_loss(W_conv6)+tf.nn.l2_loss(W_conv7)+tf.nn.l2_loss(W_out)
#loss += reg*regularizers

# LOSS AND OPTIMIZERS
batch = tf.Variable(0,dtype=tf.float32)
learning_rate = tf.train.exponential_decay(1e-3,batch*batch_size,train_size,0.95)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# TEST DATA
test_op = dice_coef_loss(predict,y_image)

# INITIALIZE VARIABLES
init_op = tf.initialize_all_variables()
sess.run(init_op)

# TRAIN
print("-- Loading Training Data ")
xs = load_images("./images.npz")
ys = load_labels("./labels.npz")
trainsize = int(0.8*xs.shape[0])
train_xs = xs[:trainsize]
test_xs = xs[trainsize:]
train_ys = ys[:trainsize]
test_ys = ys[trainsize:]
print(train_xs.shape,train_ys.shape)
print(test_xs.shape,test_ys.shape)

print("-- Training U-Net")
idx = 0
while idx < train_size:
	batch_xs,batch_ys = train_xs[idx:idx+batch_size],train_ys[idx:idx+batch_size]
	pred,l = sess.run([train_op,loss],feed_dict={x_image:batch_xs,y_image:batch_ys})
	print("Training Loss: "+str(l))
	idx+=batch_size
	#if idx%(batch_size*10)==0:
	#	step = idx/batch_size+1
	#	acc = sess.run([loss],feed_dict={x_image:test_xs,y_image:test_ys})
	#	print "Step %s | Test Loss: %s" %(step,acc)