from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import os
import glob
import os

def get_img_list(imagepath, segmentpath):
    '''
    input: 
        imagepath -> path to images directory
        Segmentpath -> path to segments directory
    output: 
        list of paths to all the images and for segmentation ground-truth 
    '''
    path1 = os.getcwd() + '/' + imagepath
    path2 = os.getcwd() + '/' + segmentpath
    imglist = glob.glob(path1 + '/*.jpg')
    annotlist = glob.glob(path2 + '/*.png')
    return imglist, annotlist

def data_preprocess(imglist, annotlist, session, height=224, width=224, num_classes=2):
    '''
    input:
        imglist, annotlist -> list of paths to all images, groung-truth
    output:
        Data -> 4-d array [number of images , height , width , 3]
        Label -> 4-d array [number of annotations , height , width , 1]
        LabelOneHot -> 4-d array [number of annotations , height , width , num_classes]
    '''
    Data = None
    Label = None
    LabelOneHot = None 
    for (f1, f2, i) in zip(imglist, annotlist, range(len(imglist))):
        # image
        img1 = Image.open(f1)
        img1 = img1.resize((height, width))
        rgb  = np.array(img1).reshape(1, height, width, 3)
        # label
        img2 = Image.open(f2)
        img2 = img2.resize((height, width), Image.LINEAR)
        label = np.array(img2).reshape(1, height, width, 1)
        # Stack images and labels
        if i == 0: 
            Data = rgb
            Label = label
        else:
            Data = np.concatenate((Data, rgb), axis=0)
            Label = np.concatenate((Label, label), axis=0)
    
    # Label for 'border' changed from '255' to '21' (new label) or to '0' (background)
    Label[Label == 255] = 0
    # remove other labels as well
    Label[Label != 0] = 1

    # Onehot-coded label
    class_labels_tensor = tf.not_equal(Label, 0) #one class for others
    background_labels_tensor = tf.equal(Label, 0) #zero class for background
    ''' for more classes we can use onehotencoding from tf.contrib (one line code) '''
    # Convert the boolean values into floats -- so that
    # computations in cross-entropy loss is correct
    bit_mask_class = tf.to_float(class_labels_tensor)
    bit_mask_background = tf.to_float(background_labels_tensor)

    LabelOneHot = session.run(tf.concat(axis=3, values=[bit_mask_class,
                                             bit_mask_background]))
    return Data, Label, LabelOneHot

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name, dtype=tf.float32)
    
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name, dtype=tf.float32)

def conv_layer(x_input, w, b, strides=[1, 1, 1, 1], padding='SAME',name='conv_'):
    with tf.name_scope(name):
        conv = tf.nn.conv2d(x_input, w, strides=strides, padding=padding)
        act = tf.nn.sigmoid(conv + b)
        tf.summary.histogram('conv/weights', w)
        tf.summary.histogram('conv/biases', b)
        tf.summary.histogram('conv/activation', act)
        return act

def deconv_layer(x_input, w, b, strides=[1, 1, 1, 1], padding='SAME',name='deconv_'):
    with tf.name_scope(name):
        shape = tf.shape(x_input)
        out_shape = [shape[0], shape[1], shape[2], w.get_shape().as_list()[2]]
        deconv = tf.nn.conv2d_transpose(x_input, filter=w, output_shape=out_shape, strides=strides, padding=padding)    
        act = tf.nn.sigmoid(deconv + b)
        tf.summary.histogram('deconv/weights', w)
        tf.summary.histogram('deconv/biases', b)
        tf.summary.histogram('deconv/activation', act)
        return act

def pool_layer(x_input):
    pool_maps, pool_argmax = tf.nn.max_pool_with_argmax(x_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return pool_maps, pool_argmax

def unravel_argmax(argmax, shape):
    output_list = [argmax // (shape[2]*shape[3]), argmax % (shape[2]*shape[3]) // shape[3]]
    return tf.stack(output_list)

def unpool_layer2x2_batch(bottom, argmax):
    bottom_shape = tf.shape(bottom)
    top_shape = [bottom_shape[0], bottom_shape[1]*2, bottom_shape[2]*2, bottom_shape[3]]

    batch_size = top_shape[0]
    height = top_shape[1]
    width = top_shape[2]
    channels = top_shape[3]

    argmax_shape = tf.to_int64([batch_size, height, width, channels])
    argmax = unravel_argmax(argmax, argmax_shape)

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [batch_size*(width//2)*(height//2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, batch_size, height//2, width//2, 1])
    t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

    t2 = tf.to_int64(tf.range(batch_size))
    t2 = tf.tile(t2, [channels*(width//2)*(height//2)])
    t2 = tf.reshape(t2, [-1, batch_size])
    t2 = tf.transpose(t2, perm=[1, 0])
    t2 = tf.reshape(t2, [batch_size, channels, height//2, width//2, 1])

    t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

    t = tf.concat([t2, t3, t1], 4)
    indices = tf.reshape(t, [(height//2)*(width//2)*channels*batch_size, 4])

    x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
    unpool_maps = tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))
    return unpool_maps

# Write figure to disk
def save_figure(plot_dir, epoch, name_prefix, img_x, img_y, img_pred, img_err):
    filename = name_prefix + str(epoch)
    plt.figure(figsize=(7, 7)) 
    plt.subplot(2, 2, 1); plt.imshow(img_x); plt.title('Input')
    plt.subplot(2, 2, 2); plt.imshow(img_y, cmap='gray'); plt.title('Ground truth')
    plt.subplot(2, 2, 3); plt.imshow(img_pred, cmap='gray'); plt.title('[Train] Prediction')
    plt.subplot(2, 2, 4); plt.imshow(np.abs(img_err) > 0.5); plt.title('Error')
    plt.savefig(os.path.join(PLOT_DIR, '{}.png'.format(filename)), bbox_inches='tight')
    
# Generate sample results from test and train data
def generate_samples(epoch, data, dataLabel_onehot, session, height, width):
    # Feed data
    index = np.random.randint(data.shape[0])
    batchData = data[index:index+1]
    batchLabel = dataLabel_onehot[index:index+1]
    # Process Data
    predMaxOut, yMaxOut = session.run([predmax, ymax], feed_dict={x: batchData, y: batchLabel})
    # Prepare Output
    refimg = data[index, :, :, :].reshape(height, width, 3)
    gtimg = yMaxOut[0, :, :].reshape(height, width)
    predimg = predMaxOut[0, :, :].reshape(height, width)
    errimg = gtimg - predimg
    return refimg, gtimg, predimg, errimg
