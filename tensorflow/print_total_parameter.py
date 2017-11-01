import tensorflow as tf
import numpy as np

def case1():
    cc = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    print('total parameter :', cc)

    sess.close()

def case2():
    # All trainable variables
    var_list1 = tf.trainable_variables()

    # Global scope trainable variables
    # var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')

    total_parameters = 0
    for variable in var_list1:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        print(variable, ":", format(variable_parameters, ','))
        total_parameters += variable_parameters
    print('Total parameter :', format(total_parameters, ','))


# main
aa = np.random.random([1, 5])
bb = [5]

X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placeholder(tf.float32, shape=[None])

net = tf.layers.dense(inputs=X, units=20)
pred = tf.layers.dense(inputs=net, units=1)

cost = tf.reduce_mean(tf.square(pred - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

case1()
case2()

