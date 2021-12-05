# CLASS DEFINITIONS FOR NEURAL NETWORKS USED IN DEEP GALERKIN METHOD

#%% import needed packages
import tensorflow as tf
import time
#%% LSTM-like layer used in DGM - modification of Keras layer class
class LSTMLayer(tf.keras.layers.Layer):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, output_dim, input_dim, trans1 = "tanh", trans2 = "tanh",strname=''):
        '''
        Args:
            input_dim (int):       dimensionality of input data
            output_dim (int):      number of outputs for LSTM layers
            trans1, trans2 (str):  activation functions used inside the layer; 
                                   one of: "tanh" (default), "relu" or "sigmoid"
        
        Returns: customized Keras layer object used as intermediate layers in DGM
        '''        
        
        # create an instance of a Layer object (call initialize function of superclass of LSTMLayer)
        super(LSTMLayer, self).__init__()
        
        # add properties for layer including activation functions used inside the layer  
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.strname = strname
        if trans1 == "tanh":
            self.trans1 = tf.nn.tanh
        elif trans1 == "relu":
            self.trans1 = tf.nn.relu
        elif trans1 == "sigmoid":
            self.trans1 = tf.nn.sigmoid
        
        if trans2 == "tanh":
            self.trans2 = tf.nn.tanh
        elif trans2 == "relu":
            self.trans2 = tf.nn.relu
        elif trans2 == "sigmoid":
            self.trans2 = tf.nn.relu
        
        ### define LSTM layer parameters (using Xavier initialization)
        # u matrix (weighting vectors for inputs original inputs x)
        self.Uz = self.add_variable("Uz", shape=[self.input_dim, self.output_dim],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Ug = self.add_variable("Ug", shape=[self.input_dim ,self.output_dim],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Ur = self.add_variable("Ur", shape=[self.input_dim, self.output_dim],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Uh = self.add_variable("Uh", shape=[self.input_dim, self.output_dim],
                                    initializer = tf.contrib.layers.xavier_initializer())
        
        # w matrix (weighting vectors for output of previous layer)        
        self.Wz = self.add_variable("Wz", shape=[self.output_dim, self.output_dim],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Wg = self.add_variable("Wg", shape=[self.output_dim, self.output_dim],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Wr = self.add_variable("Wr", shape=[self.output_dim, self.output_dim],
                                    initializer = tf.contrib.layers.xavier_initializer())
        self.Wh = self.add_variable("Wh", shape=[self.output_dim, self.output_dim],
                                    initializer = tf.contrib.layers.xavier_initializer())
        
        # bias vectors
        self.bz = self.add_variable("bz", shape=[1, self.output_dim])
        self.bg = self.add_variable("bg", shape=[1, self.output_dim])
        self.br = self.add_variable("br", shape=[1, self.output_dim])
        self.bh = self.add_variable("bh", shape=[1, self.output_dim])
    
    
    # main function to be called 
    def call(self, S, X): #foward pass of LSTM Layer
        '''Compute output of a LSTMLayer for a given inputs S,X .    
        Args:            
            S: output of previous layer
            X: data input
        
        Returns: S_new -> 
        '''   
        
        # compute components of LSTM layer output (note H uses a separate activation function)
        Z = self.trans1(tf.add(tf.add(tf.matmul(X,self.Uz, name='Matmul_Uz_'+self.strname), tf.matmul(S, self.Wz, name='Matmul_Wz_'+self.strname)), self.bz, name='ADD_bz_'+self.strname), name='trans_Z_'+self.strname)
        G = self.trans1(tf.add(tf.add(tf.matmul(X,self.Ug, name='Matmul_Ug_'+self.strname), tf.matmul(S, self.Wg, name='Matmul_Wg_'+self.strname)), self.bg, name='ADD_bg_'+self.strname), name='trans_G_'+self.strname)
        R = self.trans1(tf.add(tf.add(tf.matmul(X,self.Ur, name='Matmul_Ur_'+self.strname), tf.matmul(S, self.Wr, name='Matmul_Wr_'+self.strname)), self.br, name='ADD_br_'+self.strname), name='trans_R_'+self.strname)
        
        H = self.trans2(tf.add(tf.add(tf.matmul(X,self.Uh, name='Matmul_Uh_'+self.strname), tf.matmul(tf.multiply(S, R, name='Hadamard_SR_'+self.strname), self.Wh, name='Matmul_Wh_'+self.strname)), self.bh, name='ADD_bh_'+self.strname), name='trans_H_'+self.strname)
        
        # compute LSTM layer output
        S_new = tf.add(tf.multiply(tf.subtract(tf.ones_like(G), G, name='Sub_'+self.strname), H, name='Hadamard_H_'+self.strname), tf.multiply(Z,S, name='Hadamard_ZS_'+self.strname), name='Add_S_new_'+self.strname)
        
        return S_new

#%% Fully connected (dense) layer - modification of Keras layer class
   
class DenseLayer(tf.keras.layers.Layer):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, output_dim, input_dim, transformation=None, strname=''):
        '''
        Args:
            input_dim:       dimensionality of input data
            output_dim:      number of outputs for dense layer
            transformation:  activation function used inside the layer; using
                             None is equivalent to the identity map 
        
        Returns: customized Keras (fully connected) layer object 
        '''        
        
        # create an instance of a Layer object (call initialize function of superclass of DenseLayer)
        super(DenseLayer,self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.strname=strname
        
        ### define dense layer parameters (use Xavier initialization)
        # w vectors (weighting vectors for output of previous layer)
        self.W = self.add_variable("W", shape=[self.input_dim, self.output_dim],
                                   initializer = tf.contrib.layers.xavier_initializer())
        
        # bias vectors
        self.b = self.add_variable("b", shape=[1, self.output_dim]) # b: row vector
        
        if transformation:
            if transformation == "tanh":
                self.transformation = tf.tanh
            elif transformation == "relu":
                self.transformation = tf.nn.relu
        else:
            self.transformation = transformation
    
    
    # main function to be called 
    def call(self,X):   #foward pass of fully-connected layer
        '''Compute output of a dense layer for a given input X 
        Args:                        
            X: input to layer            
        '''
        
        # compute dense layer output
        S = tf.add(tf.matmul(X, self.W, name='Matmul_'+self.strname), self.b, name='Add_'+self.strname)  # X*W+b -> row vector -> add support broadcasting
                
        if self.transformation:
            S = self.transformation(S, name='Trans_'+self.strname)    # activation function
        
        return S

#%% Neural network architecture used in DGM - modification of Keras Model class
    
class DGMNet(tf.keras.Model):
    
    # constructor/initializer function (automatically called when new instance of class is created)
    def __init__(self, layer_width, n_layers, input_dim, final_trans=None):
        '''
        Args:
            layer_width: 
            n_layers:    number of intermediate LSTM layers
            input_dim:   spaital dimension of input data (EXCLUDES time dimension)
            final_trans: transformation used in final layer
        
        Returns: customized Keras model object representing DGM neural network
        '''  
        
        # create an instance of a Model object (call initialize function of superclass of DGMNet)
        super(DGMNet,self).__init__()
        
        # define initial layer as fully connected 
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.initial_layer = DenseLayer(layer_width, input_dim+1, transformation = "tanh", strname='1_st_DL')  
        
        # define intermediate LSTM layers
        self.n_layers = n_layers
        self.LSTMLayerList = []
                
        for _ in range(self.n_layers):
            strname = str(_+1)+'_th_DGM_layer'
            print(strname)
            self.LSTMLayerList.append(LSTMLayer(layer_width, input_dim+1, strname=strname))
        
        # define final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(1, layer_width, transformation = final_trans, strname='Final_DL')
        self.dur = [0,0,0,0,0]
    
    # main function to be called  
    def inference(self,t,x): # foward pass of the  DGM model
        '''            
        Args:
            t: sampled time inputs 
            x: sampled space inputs
        Run the DGM model and obtain fitted function value at the inputs (t,x)                
        '''  
        #start = time.time()
        with tf.device('device:GPU:0'):
        # define input vector as time-space pairs
            X = tf.concat([t,x],1) # 沿 axis1 作合併 -> size = "arbitrary"x 2
        # call initial fully connected layer
            S = self.initial_layer.call(X)
        #end = time.time()
        #self.dur[0] = end-start 
        with tf.device('device:GPU:0'):
        # call (intermediate) LSTM layers
            for i in range(self.n_layers):
                #start = time.time()
                S = self.LSTMLayerList[i].call(S,X)
                #end = time.time()
                #self.dur[i+1] = end-start
        #start = time.time()
        with tf.device('device:GPU:0'):
            # call final fully connected layers
            result = self.final_layer.call(S)
        #end = time.time()
        #self.dur[4] = end-start 
        return result
    
    # main function to be called 
    def call(self,t,x):         # foward pass of the  DGM model
        # define input vector as time-space pairs
        X = tf.concat([t,x],1) # 沿 axis1 作合併 -> size = "arbitrary"x 2
        
        # call initial fully connected layer
        S = self.initial_layer.call(X)  
        
        # call (intermediate) LSTM layers
        for i in range(self.n_layers):
            S = self.LSTMLayerList[i].call(S,X)
        
        # call final fully connected layers
        result = self.final_layer.call(S)
        
        return result
