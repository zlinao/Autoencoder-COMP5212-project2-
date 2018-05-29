import numpy as np
import argparse
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pickle
NUM_LABELS = 47
rnd = np.random.RandomState(123)
tf.set_random_seed(123)

# Following functions are helper functions that you can feel free to change
def convert_image_data_to_float(image_raw):
    img_float = tf.expand_dims(tf.cast(image_raw, tf.float32) / 255, axis=-1)
    #img_float =tf.cast(image_raw, tf.float32)
    return img_float


def visualize_ae(i, x, feat1,feat2,feat3, reconstructed_image):
    '''
    This might be helpful for visualizing your autoencoder outputs
    :param i: index
    :param x: original data
    :param features: feature maps
    :param reconstructed_image: autoencoder output
    :return:
    '''
    x = x*255
    reconstructed_image = reconstructed_image*255
    plt.figure(0)
    plt.title("original picture")
    #plt.legend()
    plt.imshow(x[i, :, :], cmap="gray")
    plt.savefig('./ae/ae_ori_plot{}'.format(i)) 
    plt.figure(1)
    plt.title("reconstructed picture")
    #plt.legend()
    plt.imshow(reconstructed_image[i, :, :, 0], cmap="gray")
    plt.savefig('./ae/ae_re_plot{}'.format(i))
    '''
    plt.figure(2)
    plt.title("first_layer feature map")
    #plt.legend()
    plt.imshow(np.reshape(feat1[i, :, :, :], (7, -1), order="F"), cmap="gray")
    plt.savefig('./ae/ae_fea1_plot{}'.format(i))
    plt.figure(3)
    plt.title("second_layer feature map")
    #plt.legend()
    plt.imshow(np.reshape(feat2[i, :, :, :], (7, -1), order="F"), cmap="gray")
    plt.savefig('./ae/ae_fea2_plot{}'.format(i))
    '''
    plt.figure(2)
    plt.title("third_layer feature map")
    #plt.legend()
    plt.imshow(np.reshape(feat3[i, :, :, :], (7, -1), order="F"), cmap="gray")
    plt.savefig('./ae/ae_fea3_plot{}'.format(i))    
def build_cnn_model(placeholder_x,placeholder_y,mode,lr,momentum,training = True):
    with tf.variable_scope("cnn"+mode) as scope:
        img_float = convert_image_data_to_float(placeholder_x)
        with tf.variable_scope('conv1') as scope:
            conv1 = tf.layers.conv2d(inputs = img_float,filters = 32, kernel_size = 3, strides=[1,1], padding='same', data_format='channels_last', 
                         activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer(seed = 123),bias_initializer=tf.zeros_initializer())
        with tf.variable_scope('conv2') as scope:
            conv2 = tf.layers.conv2d(inputs = conv1,filters = 32, kernel_size = 5, strides=(2,2), padding='same', data_format='channels_last',
                         activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer(seed = 321),bias_initializer=tf.zeros_initializer())
        with tf.variable_scope('conv3') as scope:
            conv3 = tf.layers.conv2d(inputs = conv2,filters = 64, kernel_size = 3, strides=(1,1), padding='same', data_format='channels_last',
                         activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer(seed = 231),bias_initializer=tf.zeros_initializer())
        with tf.variable_scope('conv4') as scope:
            conv4 = tf.layers.conv2d(inputs = conv3,filters = 64, kernel_size = 5, strides=(2,2), padding='same', data_format='channels_last',
                         activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer(seed = 312),bias_initializer=tf.zeros_initializer())    
       
        with tf.variable_scope('fc') as scope:
            
            conv_flattened = tf.reshape(conv4,[-1,7*7*64])
            weight1= tf.get_variable("fc1_weight",shape=(7*7*64,1024),
                                 initializer=tf.random_normal_initializer(stddev=0.01))
            hidden_layer = tf.matmul(conv_flattened, weight1)
            h_layer = tf.nn.relu(hidden_layer)
            weight2= tf.get_variable("fc2_weight",shape=(1024,NUM_LABELS),
                                   initializer=tf.random_normal_initializer(stddev=0.01))      
            logits = tf.matmul(h_layer, weight2)
																	
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=placeholder_y, logits=logits)
        if training:
            #manual momentum function:
            
            params = tf.trainable_variables()
            grads = tf.gradients(loss, params)
            accumulations= []
            new_grads = []
            new_params = []
            i = 0
            for id in range(len(grads)):
                grad = grads[i]
                param = params[i]
                if grad is not None:
                    accumulations.append(tf.Variable(np.zeros(grad.shape, dtype=np.float32)))
                    new_grads.append(grad)
                    new_params.append(param)
                i+=1
            grads = new_grads
            params = new_params
            
            var_updates = []
            for grad, accumulation, var in zip(grads,accumulations, params):
                var_updates.append(tf.assign(accumulation, momentum * accumulation + grad))
                var_updates.append(tf.assign_add(var, -lr*accumulation))
            train_op = tf.group(*var_updates)
            """ 
            optimizer = tf.train.MomentumOptimizer(momentum = momentum,learning_rate=lr)
            train_op = optimizer.minimize(loss = loss)
            """
     
        if training:
            return train_op, logits,loss
        else:
            return logits, loss

def plot_loss_acc(loss=None,val_loss=None,train_acc=None,val_acc=None,i=None,model=None):
    if model=='ae':
        plt.figure(1+3*i)
        plt.plot(loss,label='loss per epoch')
        plt.title("model"+str(i+1)+" training loss")
        plt.legend()
        plt.xlabel('epoch_num')
        plt.savefig('./ae/loss_train'+str(i))
        plt.figure(2+3*i)
        plt.plot(loss,label='loss per epoch')
        plt.title("model"+str(i+1)+" validation loss")
        plt.legend()
        plt.xlabel('epoch_num')
        plt.savefig('./ae/loss_train'+str(i))
    if model=='cnn':
        plt.figure(1+3*i)    
        plt.plot(loss,label='loss per epoch')
        plt.title("model"+str(i+1)+" training loss")
        plt.legend()
        plt.xlabel('epoch_num')
        plt.savefig('./cnn/loss_train'+str(i))
        plt.figure(2+3*i)
        plt.plot(train_acc,color='orange',label='avg accuray')
        plt.title("model"+str(i+1)+" training accuracy")
        plt.legend()
        plt.xlabel('epoch_num')
        plt.savefig('./cnn/acc_train'+str(i))
        plt.figure(3+3*i)
        plt.plot(val_acc,color='orange',label='avg accuray')
        plt.title("model"+str(i+1)+" validation accuracy")
        plt.legend()
        plt.xlabel('epoch_num')
        plt.savefig('./cnn/acc_val'+str(i))
def accuracy(logits,labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),tf.cast(labels,tf.int64)),dtype=tf.float32))
# Major interfaces
def train_cnn(x, y,x_val,y_val, placeholder_x, placeholder_y, CNN_MODEL_PATH):
    
    #grid search
    batch_size1=[20,40]
    momentum1 = [0.3,0.8]
    lr1 = [0.02,0.04]
    config = {'batch_size':0,'momentum':0,'lr':0}
    best = 0
    mode = 0
    
    
    
    time_log = []
    for batch_size in batch_size1:
        for momentum in momentum1:
            for lr in lr1:
                start_time = time.time()
                mode+=1
                m = str(mode)
                acc_log =[]
                loss_log = []
                val_log = []
                num_iterations = 100

                
                train_op,logits, loss = build_cnn_model(placeholder_x, placeholder_y,mode=m,lr=lr,momentum=momentum)
                acc = accuracy(logits,placeholder_y)
                best_acc = 0
                count=0
                cnn_saver = tf.train.Saver()
                l = np.arange(len(x))
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    
                    for n_pass in range(num_iterations):
                        #shuffle training set before breaking
                        loss_list=[]
                        acc_list = []
                        np.random.shuffle(l)
                        x = x[l]
                        y = y[l]
                        #break training set into minibatch
                        for i in range(len(x)//batch_size):
                            '''
                            if i == len(x)//batch_size:
                                x_batch = x[i*batch_size:]
                                y_batch = y[i*batch_size:]
                            
                            else:
                            '''
                            x_batch = x[i*batch_size:(i+1)*batch_size]
                            y_batch = y[i*batch_size:(i+1)*batch_size]
                            
                            
                            feed_dict = {placeholder_x: x_batch, placeholder_y: y_batch}
                            _,loss1,acc1 = sess.run([train_op,loss,acc], feed_dict=feed_dict)
                            #print('batch accuracy:{}',format(acc1))
                            loss_list.append(loss1)
                            acc_list.append(acc1)
                        print("Epoch {} finished".format(n_pass))
                        #print(acc_list)
                        loss_avg = np.sum(loss_list)/len(loss_list)
                        acc_avg = np.sum(acc_list)/len(acc_list)
                        loss_log.append(loss_avg)
                        acc_log.append(acc_avg)
                        print('trianing accuracy: {}'.format(acc_avg))
                        print('trianing loss: {}'.format(loss_avg))
                        if n_pass%1==0:
                            
                            feed_dict = {placeholder_x: x_val, placeholder_y: y_val}
                            acc1 = sess.run(acc, feed_dict=feed_dict)
                            if acc1>best_acc:
                                best_acc=acc1
                            else:
                                count+=1
                            print('validation accuracy: {}'.format(acc1))
                            val_log.append(best_acc)
                            if count>=5:
                                plot_loss_acc(loss = loss_log,train_acc=acc_log,val_acc=val_log,i=mode,model='cnn')
                                if best<best_acc:
                                    best = best_acc
                                    config['batch_size'] =batch_size
                                    config['momentum'] =momentum
                                    config['lr']=lr
                                    config['acc'] = best
                                    config['mode'] = mode
                                    cnn_saver.save(sess=sess, save_path=CNN_MODEL_PATH)
                                break
                cost_time = time.time()-start_time
                time_log.append(cost_time)
    print('best hyperparameter:')
    print('batch_size:',config['batch_size'])
    print('momentum:',config['momentum'])
    print('learning rate:',config['lr'])
    print('validation accuracy:',config['acc'])
    print('average training time:',sum(time_log)/len(time_log))
    with open('config.pkl', 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)


def test_cnn(x, y, CNN_MODEL_PATH):
    with open('config.pkl', 'rb') as f:
        config = pickle.load(f, encoding='utf-8')
    with tf.Graph().as_default():
        placeholder_x = tf.placeholder(tf.uint8, shape=(None, 28, 28), name="imag")
        placeholder_y = tf.placeholder(tf.int32, shape=(None,), name="label")
        logits,loss = build_cnn_model(placeholder_x, placeholder_y,mode=str(config['mode']),lr = config['lr'],momentum=config['momentum'], training=False)
        acc = accuracy(logits,placeholder_y)
        cnn_saver = tf.train.Saver()
        acc1=0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            cnn_saver.restore(sess,CNN_MODEL_PATH)
            feed_dict = {placeholder_x: x, placeholder_y: y}
            acc1,loss1 = sess.run([acc,loss],feed_dict=feed_dict)
            print('Test accuracy: {}'.format(acc1))
            print('Test loss: {}'.format(loss1))
        return acc1

def ae_model(placeholder_x,mode,lr,momentum,training = True):
    with tf.variable_scope("ae"+mode) as scope:
        img_float = convert_image_data_to_float(placeholder_x)
        with tf.variable_scope('conv1') as scope:
            conv1 = tf.layers.conv2d(inputs = img_float,filters = 32, kernel_size = 5, strides=[2,2], padding='same', data_format='channels_last', 
                         activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer(seed = 123),bias_initializer=tf.zeros_initializer())
        with tf.variable_scope('conv2') as scope:
            conv2 = tf.layers.conv2d(inputs = conv1,filters = 64, kernel_size = 5, strides=(2,2), padding='same', data_format='channels_last',
                         activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer(seed = 321),bias_initializer=tf.zeros_initializer())
        with tf.variable_scope('conv3') as scope:
            conv3 = tf.layers.conv2d(inputs = conv2,filters = 2, kernel_size = 3, strides=(1,1), padding='same', data_format='channels_last',
                         activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer(seed = 231),bias_initializer=tf.zeros_initializer())
        with tf.variable_scope('dconv1') as scope:
            dconv1 = tf.layers.conv2d_transpose(inputs = conv3,filters = 2, kernel_size = 3, strides=(1,1), padding='same', data_format='channels_last',
                         activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer(seed = 231),bias_initializer=tf.zeros_initializer())
        with tf.variable_scope('dconv2') as scope:
            dconv2 = tf.layers.conv2d_transpose(inputs = dconv1,filters = 64, kernel_size = 5, strides=(2,2), padding='same', data_format='channels_last',
                         activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer(seed = 321),bias_initializer=tf.zeros_initializer())
        with tf.variable_scope('dconv3') as scope:
            dconv3 = tf.layers.conv2d_transpose(inputs = dconv2,filters = 32, kernel_size = 5, strides=[2,2], padding='same', data_format='channels_last', 
                         activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer(seed = 123),bias_initializer=tf.zeros_initializer())
        img = tf.identity(dconv3)
        params = tf.trainable_variables()
        loss = 0
        mse_loss = 0 
        regularizer = 0
        with tf.variable_scope('train_loss') as scope:
            for p in params:
                regularizer += 0.01*tf.reduce_mean(tf.square(p))
            mse_loss += tf.div(tf.reduce_mean(tf.square(tf.subtract(img, img_float))),2,name="mse")
            loss = mse_loss+regularizer
        
            
            #manual momentum function:
            
            params = tf.trainable_variables()
            grads = tf.gradients(loss, params)
            accumulations= []
            new_grads = []
            new_params = []
            i = 0
            for id in range(len(grads)):
                grad = grads[i]
                param = params[i]
                if grad is not None:
                    accumulations.append(tf.Variable(np.zeros(grad.shape, dtype=np.float32)))
                    new_grads.append(grad)
                    new_params.append(param)
                i+=1
            grads = new_grads
            params = new_params
            
            var_updates = []
            for grad, accumulation, var in zip(grads,accumulations, params):
                var_updates.append(tf.assign(accumulation, momentum * accumulation + grad))
                var_updates.append(tf.assign_add(var, -lr*accumulation))
            train_op = tf.group(*var_updates)
     
        if training:
            return train_op, loss ,mse_loss
        else:
            return img_float, img, conv1,conv2,conv3
def train_ae(x, x_val, placeholder_x, ae_MODEL_PATH):
    # TODO: implement autoencoder training
    #grid search
    batch_size1=[20,40]
    momentum1 = [0.2,0.9]
    lr1 = [0.1,0.5]
    config = {'batch_size':0,'momentum':0,'lr':0}
    best = 1000000000
    mode = 0
    time_log = []
    
    
    for batch_size in batch_size1:
        for momentum in momentum1:
            for lr in lr1:
                start_time = time.time()
                mode+=1
                m = str(mode)
                loss_log = []
                val_log = [] 
                num_iterations = 100

                
                train_op, loss, mse_loss = ae_model(placeholder_x,mode =m,lr = lr,momentum = momentum,training = True)
                
                
                min_loss = 100000
                count=0
                cnn_saver = tf.train.Saver()
                l = np.arange(len(x))
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    
                    for n_pass in range(num_iterations):
                        #shuffle training set before breaking
                        loss_list=[]
                        
                        np.random.shuffle(l)
                        x = x[l]
                        
                        #break training set into minibatch
                        for i in range(len(x)//batch_size):
                           
                            x_batch = x[i*batch_size:(i+1)*batch_size]
                            
                            
                            
                            feed_dict = {placeholder_x: x_batch}
                            _,loss1 = sess.run([train_op,loss], feed_dict=feed_dict)
                            #print('batch accuracy:{}',format(acc1))
                            loss_list.append(loss1)
                            
                        print("Epoch {} finished".format(n_pass))
                        #print(acc_list)
                        loss_avg = np.sum(loss_list)/len(loss_list)
                        
                        loss_log.append(loss_avg)
                        
                        
                        print('trianing loss: {}'.format(loss_avg))
                        if n_pass%1==0:
                            
                            feed_dict = {placeholder_x: x_val}
                            val_loss = sess.run(mse_loss, feed_dict=feed_dict)
                            if val_loss < min_loss:
                                min_loss = val_loss
                            else:
                                count+=1
                            print('validation loss: {}'.format(val_loss))
                            val_log.append(val_loss)
                            if count>=2:
                                plot_loss_acc(loss = loss_log,val_loss = val_log,i=mode,model='ae')
                                if best>min_loss:
                                    best = min_loss
                                    config['batch_size'] =batch_size
                                    config['momentum'] =momentum
                                    config['lr']=lr
                                    config['loss'] = best
                                    config['mode'] = mode
                                    cnn_saver.save(sess=sess, save_path=ae_MODEL_PATH)
                                break
                cost_time = time.time()-start_time
                time_log.append(cost_time)            
    print('best hyperparameter:')
    print('batch_size:',config['batch_size'])
    print('momentum:',config['momentum'])
    print('learning rate:',config['lr'])
    print('validation loss:',config['loss'])
    print('average training time:',sum(time_log)/len(time_log))
    with open('config_ae.pkl', 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate_ae(x,placeholder_x,ae_MODEL_PATH='./trained_ae_model.ckpt'):
    with open('config_ae.pkl', 'rb') as f:
        config = pickle.load(f, encoding='utf-8')
    with tf.Graph().as_default():
        placeholder_x = tf.placeholder(tf.uint8, shape=(None, 28, 28), name="image")
        
        img_float,image,fea1,fea2,fea3 = ae_model(placeholder_x, mode=str(config['mode']),lr = config['lr'],momentum=config['momentum'], training=False)
        
        cnn_saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            cnn_saver.restore(sess,ae_MODEL_PATH)
            feed_dict = {placeholder_x: x}
            x, reconstructed_image,feature1,feature2,feature3 = sess.run([img_float,image,fea1,fea2,fea3], feed_dict=feed_dict)
            for ii in range(3):
                visualize_ae(i=ii, x =np.squeeze(x), feat1 = np.squeeze(feature1),feat2 = np.squeeze(feature2),feat3 = np.squeeze(feature3), reconstructed_image=np.squeeze(reconstructed_image))
            
        


def main():
    parser = argparse.ArgumentParser(description='COMP5212 Programming Project 2')
    parser.add_argument('--task', default="train_ae", type=str,
                        help='Select the task, train_cnn, test_cnn, '
                             'train_ae, evaluate_ae, ')
    parser.add_argument('--datapath',default="./datasets",type=str, required=False,
                        help='Select the path to the data directory')
    args = parser.parse_args()
    datapath = args.datapath
    with tf.variable_scope("placeholders"):
        img_var = tf.placeholder(tf.uint8, shape=(None, 28, 28), name="img")
        label_var = tf.placeholder(tf.int32, shape=(None,), name="true_label")

    if args.task == "train_cnn":
        file_train = np.load(datapath+"/data_classifier_train.npz")
        x_train = file_train["x_train"]
        y_train = file_train["y_train"]

        length = int(len(x_train)*0.8)
        #break training set into trianing set and validation set
        X_train = x_train[:length]
        Y_train = y_train[:length]
        X_val = x_train[length:]
        Y_val = y_train[length:]      
        train_cnn(X_train, Y_train,X_val, Y_val, img_var, label_var,CNN_MODEL_PATH='./trained_cnn_model.ckpt')
    elif args.task == "test_cnn":
        file_test = np.load(datapath+"/data_classifier_test.npz")
        x_test = file_test["x_test"]
        y_test = file_test["y_test"]
        accuracy = test_cnn(x_test, y_test,CNN_MODEL_PATH = './trained_cnn_model.ckpt')
        
    elif args.task == "train_ae":
        file_unsupervised = np.load(datapath + "/data_autoencoder_train.npz")
        x_ae_train = file_unsupervised["x_ae_train"]
        length = int(len(x_ae_train)*0.8)
        X_train = x_ae_train[:length]
        X_val = x_ae_train[length:]
        train_ae(X_train,X_val, img_var, ae_MODEL_PATH='./trained_ae_model.ckpt')
    elif args.task == "evaluate_ae":
        file_unsupervised = np.load(datapath + "/data_autoencoder_eval.npz")
        x_ae_eval = file_unsupervised["x_ae_eval"]
        evaluate_ae(x_ae_eval, img_var,ae_MODEL_PATH='./trained_ae_model.ckpt')


if __name__ == "__main__":
    main()
