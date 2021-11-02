"""
分析DGM 模型推論程式碼
"""
import DGM
import FokkerPlanck_OUwithGaussianStart as FPOU
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from tensorflow.python.client import timeline
plt.close('all')

tf.reset_default_graph()   # To clear the defined variables and operations of the previous call
 
#%% Parameter Setting
# OU process parameters 
kappa = 0.5  # mean reversion rate
theta = 0.0  # mean reversion level
sigma = 2    # volatility

# mean and standard deviation for (normally distributed) process starting value
alpha = 0.0
beta = 1

# terminal time 
T = 1.0

# bounds of sampling region for space dimension, i.e. sampling will be done on
# [multiplier*Xlow, multiplier*Xhigh]
Xlow = -4.0
Xhigh = 4.0
x_multiplier = 2.0
t_multiplier = 1.5

# neural network parameters
num_layers = 3
nodes_per_layer = 1000        # neural network output dimension
learning_rate = 0.001

# Training parameters
sampling_stages = 50    # number of times to resample new time-space domain points
steps_per_sample = 10    # number of SGD steps to take before re-sampling

# Sampling parameters
nSim_t = 5               # the number of sampling time points
nSim_x_interior = 50     # the number of sampling space points at each sampling time points 
nSim_x_initial = 50      # the number of sampling space points at initial-time points 

#%% Construct the model
model = DGM.DGMNet(nodes_per_layer, num_layers, 1)
sess = tf.Session()
"""Setting tensor object for input data / loss  of DGM model"""
# input tensor
t_tnsr = tf.placeholder(tf.float32, [None,1])
x_interior_tnsr = tf.placeholder(tf.float32, [None,1])
x_initial_tnsr = tf.placeholder(tf.float32, [None,1])
"""define our output u of DGM model """
u = model(t_tnsr, x_interior_tnsr)
p_unnorm = tf.exp(-u)
"""Write event file of computational graph"""
writer = tf.summary.FileWriter('./graphs')
writer.add_graph(p_unnorm.graph) #tf.compat.v1.get_default_graph())
writer.flush()
# loss tensor
L1_tnsr, L3_tnsr = FPOU.loss(model, t_tnsr, x_interior_tnsr, x_initial_tnsr, nSim_t, alpha, beta)
loss_tnsr = L1_tnsr + L3_tnsr
# set optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_tnsr)

"""After network setting, need to initialize variables"""
init_op = tf.global_variables_initializer()
sess.run(init_op)

"""runing option for profiling"""
run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
run_metadata = tf.compat.v1.RunMetadata()

#%% Train the network

# Start training the DGM network
loss_mat = np.zeros((sampling_stages,1))  # store the total loss
for i in range(sampling_stages):        # sampling_stages: 50
    
    # sample uniformly from the required regions
    t, x_interior, x_initial = FPOU.sampler(nSim_t, nSim_x_interior, nSim_x_initial)
    
    # for  given samples, take the required number of SGD steps
    for j in range(steps_per_sample):   # #steps_per_sample: 10
        loss,L1,L3,_ = sess.run([loss_tnsr, L1_tnsr, L3_tnsr, optimizer],
                                feed_dict = {t_tnsr:t, x_interior_tnsr:x_interior, x_initial_tnsr:x_initial},
                                )
    loss_mat[i] = loss
    print(i)
"""
#%% Profile the training of model
# define profile setting
profile_opts = tf.compat.v1.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.time_and_memory()).select(["bytes", "micros", "params", "float_ops"]).with_file_output("profiler.txt").with_timeline_output("timeline_Train.json").order_by("micros") .build()
# run profile under graph p_unnorm
tf.compat.v1.profiler.profile(
    graph= tf.compat.v1.get_default_graph(), # graph
    run_meta= run_metadata,  #  previous run_metadata 
    cmd='graph',
    options=profile_opts
)
"""
#%% Run the inference of DGM model
# run just one inference per time profiling
"""
# Input parameter setting
densityTimes = [0,0.1*T, 0.25*T, 0.5*T,0.75*T, T]
x_plot = np.linspace(Xlow*x_multiplier, Xhigh*x_multiplier, 1000)
for i, curr_t in enumerate(densityTimes):
    t_plot = curr_t * np.ones_like(x_plot.reshape(-1,1))
    unnorm_dens = sess.run([p_unnorm], feed_dict= {t_tnsr:t_plot, x_interior_tnsr:x_plot.reshape(-1,1)},
                           options=run_options, run_metadata=run_metadata)
"""
"""
# define profile setting
profile_opts = tf.compat.v1.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.time_and_memory()).select(["bytes", "micros", "params", "float_ops"]).with_file_output("profiler.txt").with_timeline_output("timeline_Train.json").order_by("micros") .build()
# run profile under graph p_unnorm
tf.compat.v1.profiler.profile(
    graph= tf.compat.v1.get_default_graph(), # graph
    run_meta= run_metadata,  #  previous run_metadata 
    cmd='graph',
    options=profile_opts
)
"""
# run N inferences per time profiling
#u_list = [p_unnorm for ii in range(10000)]   # list of tensor-objects
u_list = [p_unnorm, u]
x_plot = np.linspace(Xlow*x_multiplier, Xhigh*x_multiplier, 1000) # model input
t_plot = 0.5 * np.ones_like(x_plot.reshape(-1,1))                 # model input
unnorm_dens_list = sess.run(u_list, feed_dict= {t_tnsr:t_plot, x_interior_tnsr:x_plot.reshape(-1,1)},
                           options=run_options, run_metadata=run_metadata)
# Store the timeline data
tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()
with open("timeline_inference5000.json", 'w') as f:
    f.write(ctf)
