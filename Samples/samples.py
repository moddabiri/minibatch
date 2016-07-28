__author__ = "Mohammad Dabiri"
__copyright__ = "Free to use, copy and modify"
__credits__ = ["Mohammad Dabiri"]
__license__ = "MIT Licence"
__version__ = "0.0.1"
__maintainer__ = "Mohammad Dabiri"
__email__ = "moddabiri@yahoo.com"

import os, sys, inspect, importlib

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
mb_path = os.path.join(current_dir, '..')
sys.path.append(mb_path)
from MiniBatchSet import MiniBatchSet
import numpy as np

if not importlib.find_loader('tensorflow') is None:
    import tensorflow as tf
else:
    print("WARNING! Tensorflow is not installed on the vm.")

if not importlib.find_loader('sklearn') is None:
    from sklearn import linear_model
else:
    print("WARNING! sklearn is not installed on the vm.")



#*****!!NOTE!!    Sample configs, change these before running the sample
snapshot_path = os.path.join(current_dir, "test_data/minibatchset_ss.json")
train_folder = os.path.join(current_dir, "test_data/train/")
test_folder = os.path.join(current_dir, "test_data/test/")
output_path = os.path.join(current_dir, "test_data/output/")
even_path = os.path.join(current_dir, "test_data/even/")
number_of_iterations = 10
batch_size = 10
training_kit = 'tensorflow' #'sklearn' / 'none'

def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
      return tf.Variable(initial, dtype=tf.float32)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)


#Scenario 1: Mini-batches kept on memory
#This method loads all the samples on the memory. Best if the sample can be fit on the memory.
def sample1():    
    bs = MiniBatchSet(file_paths_or_folder=train_folder, batch_size=batch_size, hold_in_memory=True)
    bs.save_snapshot(snapshot_path)
    do_train(bs)
  
#TF - Scenario 2: Mini-batches read from the files in the runtime. 
#This method keeps an index list of all the rows in the set on memory which requires more memory, but, it assures that all of the samples are visited once and only once in each round.
#It works best if you could allocate the required memory for the index list and assure completely normal distribution of the sample. Works well with medium size sets.
def sample2():    
    bs = MiniBatchSet(file_paths_or_folder=train_folder, batch_size=batch_size)
    bs.save_snapshot(snapshot_path)
    do_train(bs)  

#Scenario 3: Mini-batches read from the files with ad-hoc random sampling.
#This method is designed for the large sets. It will not store the index list to avoid memory overflow, but performs a normally distributed random sampling from the data. However, it will perform slowly in order to fetch the data from the disk in each batch request.
def sample3():    
    bs = MiniBatchSet(file_paths_or_folder=train_folder, batch_size=batch_size)
    bs.save_snapshot(snapshot_path)
    do_train(bs)  

#Scenario 4: Mini-batches read from the files with ad-hoc random sampling.
#This method is the optimized alternative for the 3rd scenario. It loads a bulk of mini-batches on the memory to avoid data I/O overhead. It fetches another batch if the loaded bulk goes lower than 50%.
def sample4():    
    bs = MiniBatchSet(file_paths_or_folder=train_folder, batch_size=batch_size, bulk_size=10)
    bs.save_snapshot(snapshot_path)
    do_train(bs)  

#Scenario 5: Restoring a snapshot.
def sample5():    
    bs = MiniBatchSet(file_paths_or_folder=train_folder, batch_size=batch_size)
    bs.save_snapshot(snapshot_path)
    bs2 = MiniBatchSet.from_snapshot(snapshot_path)
    assert bs == bs2

#Scenario 6: Normalizing a data set.
def sample6():    
    bs = MiniBatchSet(file_paths_or_folder=train_folder)   
    bs.normalize(output_path, [0,1,2,3,4])

#Scenario 7: Performing custom actions on the data.
def sample7():    
    bs = MiniBatchSet(file_paths_or_folder=train_folder)

    #----------------Step1: Get the maximum of all the features
    max_of_features = np.array([0.0] * 7,np.float64)

    #the first argument is reserved for data transformer to input the sample to the action
    #0 or more arguments could be used for running variables. Note that these variables must be returned as a list in order for the transform function to keep them running through the iterations
    def get_max_action(data, max_of_features):
            data = np.array(data, dtype=np.float64)
            max_of_features = np.maximum(max_of_features, data)
            
            #The running arguments must be returned to keep them running through the iterations
            return [max_of_features]

    #The output path is passed as None, it makes the transformer to not produce any output files and simply execute the action over the rows
    max_of_features = bs.transform_data(get_max_action, None, max_of_features)[0]

    #----------------Step2: Divide the features by their maximum values and write it to an output file
    max_of_features = np.array([1.0 if x == 0.0 else x for x in max_of_features], dtype=np.float32)
    def divide_by_max_action(data):
            data = np.array(data, dtype=np.float64)
            data = np.divide(data, max_of_features)
            return data, []

    #The output path is passed now, it will generate the transformed data in the destination
    bs.transform_data(divide_by_max_action, output_path)

#Scenario 8: Distribute the data amongst files with the same number of rows. This will help optimize the sampling process by avoiding too small files to waste the time or too large files to overflow the memory.
def sample8():    
    bs = MiniBatchSet(file_paths_or_folder=train_folder)
    bs.even_files(100, even_path)

def do_train(bs):
    if training_kit == 'tensorflow':
        train_tf(bs)
    elif training_kit == 'sklearn':
        train_sk(bs)
    else:
        print("None of the known training kits were chosen. Skipping the training.")

#A simple nn with tensorflow
def train_tf(bs):
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 6])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    W_layer1 = weight_variable([6, 64])
    b_layer1 = bias_variable([64])
    h_layer1 = tf.nn.softsign(tf.matmul(x, W_layer1) + b_layer1)

    W_readout = weight_variable([64, 1])
    b_readout = bias_variable([1])
    y_readout = tf.sigmoid(tf.matmul(h_layer1, W_readout) + b_readout)

    loss = tf.reduce_mean(tf.square(y_ - y_readout))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    sess.run(tf.initialize_all_variables())

    for i in range(number_of_iterations):
        minibatch = np.matrix(next(bs))
        batch_x = np.asarray(minibatch[:,0:6])
        batch_y = np.asarray(minibatch[:,6])

        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

#Basic linear regression with sklearn
def train_sk(bs):
    regr = linear_model.SGDRegressor()

    for i in range(number_of_iterations):
        minibatch = np.matrix(next(bs))
        batch_x = np.asarray(minibatch[:,0:6])
        batch_y = np.asarray(minibatch[:,6])
        regr.partial_fit(batch_x, batch_y)

if __name__ == '__main__':
    sample1()
    sample2()
    sample3()
    sample4()
    sample5()
    sample6()
    sample7()
    sample8()