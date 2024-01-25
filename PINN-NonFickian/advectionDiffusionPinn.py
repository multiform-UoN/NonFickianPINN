#%%
# import
#############################
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.constraints import NonNeg
from keras.regularizers import l1_l2
from time import time
import matplotlib.pyplot as plt
import os

# # Configure TensorFlow to use the specified number of threads
# tf.config.threading.set_intra_op_parallelism_threads(4)
# tf.config.threading.set_inter_op_parallelism_threads(4)
 
#%%
# USER INPUTS
#############################

## general parameters
tol = 1e-8 # tolerance for stopping training

## Data and test case
testcase = "testcase4" # Testcase (choose the one you want to run)
coarsen_data = 1 # coarsening of the data (1 = no coarsening, >1 = coarsening skipping points)
data_perturbation = 0e-2 # perturbation for the data


## Parameters
train_parameters = True # train the parameters or not
nparam = 2 # number of parameters to train (d,u,beta0) 1=only d, 2=d and u, 3=d,u and beta0
param_perturbation = 10 # perturbation for the parameters - factor for random perturbation of the parameters
learning_rate_param = 1e-2 # learning rate of the parameters
train_parameters_epoch = 1000 # epoch after which train the parameters

## Loss function weights (will be normalised afterwards)
pde_weight = 1.      # penalty for the PDE
data_weight = 1.     # penalty for the data fitting (will be multiplied by param_data_factor)
ic_weight = 10.    # penalty for the initial condition
bc_weight = 10.     # penalty for the boundary condition

# NN training parameters
epochs = 5000          # number of epochs
epoch_print = 10      # print the loss every epoch_print epochs

learning_rate = 1e-2   # learning rate for the network weights
learning_rate_decay_factor = 0.95 # decay factor for the learning rate
learning_rate_step = 100
# Piecewise constant learning rate every Y epochs decayed by X
learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    [learning_rate_step*(i+1) for i in range(int(epochs/learning_rate_step))],
    [learning_rate*learning_rate_decay_factor**i for i in range(int(epochs/learning_rate_step)+1)])
# # smooth exponential decay
# learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
#     learning_rate,
#     decay_steps=learning_rate_step,
#     decay_rate=learning_rate_decay_factor,
#     staircase=False)
# # polynomial decay
# learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
#     learning_rate,
#     decay_steps=learning_rate_step,  # Adjust the decay steps as needed
#     end_learning_rate=learning_rate*1e-3,
#     power=1.0,
#     cycle=False)

## NN architecture
num_hidden_layers = 8 # number of hidden layers (depth of the network)
num_neurons = 20      # max number of neurons per layer (width of the network)
def num_neurons_per_layer(depth): # number of neurons per layer (adapted to the depth of the network)
    return num_neurons    # constant number of neurons
    # return np.floor(num_neurons*(np.exp(-(depth-0.5)**2 * np.log(num_neurons/2.1)/((0.5)**2))))  # Gaussian distribution of neurons
activation = 'tanh' # 'sigmoid' or 'tanh'



#%%
# Load data
#############################

# Check if the folder exists
datafolder = "../data/"+testcase
if not os.path.exists(datafolder):
    # If it doesn't exist, use the folder name without the "../"
    datafolder = "data/"+testcase

# Load data as pandas dataframes
p = pd.read_csv(f'{datafolder}/p.csv', header=None, dtype=np.float32)
x_grid = pd.read_csv(f'{datafolder}/x.csv', header=None, dtype=np.float32)
t_grid = pd.read_csv(f'{datafolder}/t.csv', header=None, dtype=np.float32)
c_data = pd.read_csv(f'{datafolder}/c.csv', header=None, dtype=np.float32)
nt = np.shape(t_grid)[0]
nx = np.shape(x_grid)[0]

# perturb data randomly
# c_data = c_data + data_perturbation * np.random.randn(c_data.size).reshape(c_data.shape) # additive noise
c_data = c_data * (1 + data_perturbation * np.random.randn(c_data.size).reshape(c_data.shape)) # multiplicative noise

# reshape data
c_data_2d = np.reshape(c_data, (nt, nx))

# Coarsen data
if coarsen_data>1:
    x_grid = pd.concat([x_grid[:-1-(nx+1)%coarsen_data:coarsen_data],x_grid.iloc[-1]])
    t_grid = pd.concat([t_grid[:-1-(nt+1)%coarsen_data:coarsen_data],t_grid.iloc[-1]])
    rows = np.append(np.arange(0, nt - 1-(nt+1)%coarsen_data, coarsen_data),nt-1)
    cols = np.append(np.arange(0, nx - 1-(nx+1)%coarsen_data, coarsen_data),nx-1)
    c_data_2d = c_data_2d[rows[:, None], cols]
    # c_data_2d = c_data_2d[::coarsen_data, ::coarsen_data]
    c_data = c_data_2d.flatten()

# mesh and time discretization
nt = np.shape(t_grid)[0]
nx = np.shape(x_grid)[0]
Xgrid, Tgrid = np.meshgrid(x_grid, t_grid)

# convert data to numpy arrays and float32
p = np.array(p).squeeze().astype(np.float32)
x_data = Xgrid.flatten().astype(np.float32)
t_data = Tgrid.flatten().astype(np.float32)
c_data = c_data.astype(np.float32)

# Convert data to tensor because tf.GradientTape() can only watch tensor and not numpy arrays
x_tf = tf.expand_dims(tf.convert_to_tensor(x_data), -1)
t_tf = tf.expand_dims(tf.convert_to_tensor(t_data), -1)
c_tf = tf.convert_to_tensor(c_data)

# tf differentiation variables
tt = tf.Variable(t_tf)
xx = tf.Variable(x_tf)

# model inputs
inputs = [xx, tt, c_tf]

# additional variables added to gradient tracking
randp =  (p * param_perturbation**(np.random.rand(p.size)*2 -1 )).astype(np.float32) # perturb parameters randomly
d = tf.Variable([randp[1]], trainable=train_parameters) # diffusion coefficient
u = tf.Variable([randp[2]], trainable=(train_parameters and nparam>1))  # advection velocity
beta0 = tf.Variable([randp[0]], trainable=(train_parameters and nparam>2))  # porosity
params = [d, u, beta0]
params0 = [p[1], p[2], p[0]] # true parameters

#%%
# Define the PINN model and loss functions
#############################

def pinn_model(num_hidden_layers=num_hidden_layers, num_neurons_per_layer=num_neurons_per_layer):
    x_input = keras.Input(shape=(1,))
    t_input = keras.Input(shape=(1,))

    output_c = layers.concatenate([t_input, x_input]) # input layer
    
    # hidden layers
    for i in range(num_hidden_layers):
        output_c = tf.keras.layers.Dense(num_neurons_per_layer((i+1)/num_hidden_layers),
                                         activation=activation,  
                                        #  kernel_constraint=NonNeg(), # this gives trivial constant fitting
                                         kernel_initializer='glorot_normal',
                                        #  kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
                                         )(output_c)
    
    # output layer
    output_c = tf.keras.layers.Dense(1)(output_c)

    return keras.Model(inputs=[t_input, x_input], outputs=output_c)

@tf.function(reduce_retracing=True)
def custom_loss(inputs, model):

    # inputs
    xx, tt, cc = inputs

    # Compute derivatives:
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(tt)
        tape.watch(xx)
        # output model
        output_model = model([tt, xx])
        c_model = output_model[:, 0]
        c_model = tf.expand_dims(c_model, -1)
        # derivatives and fluxes
        c_x = tape.gradient(c_model, xx)
        c_t = tape.gradient(c_model, tt)
    div_output = u * c_x - d * tape.gradient(c_x, xx)
    del tape

    # Compute the components of loss function
    norm_weight = x_data.max()**2 / (tf.multiply(beta0,d)) # normalization factor for the PDE
    pde_loss = tf.reduce_mean(tf.multiply(norm_weight, (
        (tf.multiply(beta0, c_t) + div_output) ** 2 
        ))) # PDE loss
    data_fitting_loss = tf.reduce_mean((c_model - cc) ** 2) # data misfit
    bc_fitting_loss = tf.reduce_mean((c_model[::nx] - cc[::nx]) ** 2 # dirichlet based on data
                            + (c_x[(nx-1)::nx]) ** 2) # neumann zero
    ic_fitting_loss = tf.reduce_mean((c_model[0:nt] - cc[0:nt])  ** 2) # initial condition based on data
    
    return [pde_loss, data_fitting_loss, ic_fitting_loss, bc_fitting_loss]


#%%
# Create the PINN model
model = pinn_model()
trainable = model.trainable_variables

# add the parameters to the trainable variables
if train_parameters:
    for i in range(nparam):
        trainable.append(params[i])

#%%
# Train the NN
#############################

# Create the optimizer with a smaller learning rate
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate,epsilon=1e-08,amsgrad=True)

# some lists to save the losses and parameters
losses = np.zeros((epochs, 9))
param_values = np.zeros((epochs, nparam))
param_grads = np.zeros((epochs, nparam))
param_hessian = np.zeros((epochs, nparam))

# Start loop and timer
stop = False
t0 = time()
t1 = t0

# Epoch loop
for epoch in range(epochs):
    # compute factor for training the parameters and weighing the data
    
    # exponential transition from 0 to 1
    # param_data_factor = np.exp(-np.log(train_parameters_epoch)*(epochs-epoch)/epochs).item()
    
    # tanh transition from 0 to 1
    param_data_factor = (np.tanh(10*(epoch-epochs/2-train_parameters_epoch)/epochs)+1)/2
    
    # # sigmoid transition from 0 to 1
    # param_data_factor = 1/(1+np.exp(-10*(epoch-epochs/2-train_parameters_epoch)/epochs))
    
    # set the factor to exactly 0 for the first epochs
    param_data_factor *= (epoch>train_parameters_epoch)

    # compute the adaptive weights
    weights = [pde_weight, data_weight*param_data_factor, ic_weight, bc_weight]
    # normalise the weights
    weights = [w/sum(weights) for w in weights]
    
    # Compute the gradients
    with tf.GradientTape(persistent=True) as tape:
        # Call the tf decorated loss function
        loss0 = custom_loss(inputs, model) # unweighted loss terms
        loss = [l * w for l, w in zip(loss0, weights)] # weight the losses
        # Append the total loss
        loss.append(sum(loss)) # weighted total loss
    gradients = tape.gradient(loss[-1], trainable)
    # hessian = tape.gradient(gradients[-nparam:], trainable[-nparam:])
    del tape
    
    if (train_parameters):
        param_grads[epoch,:] = np.array(gradients[-nparam:]).squeeze()/weights[0] # store parameter gradients
    
        # param_hessian[epoch,:] = np.array(hessian[-1])/weights[0] # store parameter hessian
    
        # # Manually apply gradients to the parameters
        # for i in range(nparam):
        #     params[i].assign_sub(gradients[-nparam+i]*learning_rate_param*param_data_factor)
        
        # scale the parameter gradients
        for i in range(-nparam,0):
            gradients[i] *= learning_rate_param*param_data_factor
            print(i)
            print(gradients[-i].shape)        
    
    # # Apply the gradients to update the weights
    # optimizer.apply_gradients(zip(gradients[:-nparam], trainable[:-nparam]))

    # Apply all the gradients
    optimizer.apply_gradients(zip(gradients, trainable))
    
    # store losses (unweighted and weighted concatenated)
    losses[epoch,:] = np.array(loss0+loss) 
    # store parameter values
    param_values[epoch,:] = np.array(params[:nparam]).squeeze()
    
    # outputs to screen
    if epoch % epoch_print == 0:
        print(f"\nEpoch {epoch + 1}/{epochs}, Loss: {loss[-1].numpy()}")
        print(f"param_data_factor = {param_data_factor:.2e} beta0 = {beta0.numpy()[0]}, d = {d.numpy()[0]}, u = {u.numpy()[0]} ")
        # print('CPU time for {} epochs: {} seconds'.format(epoch_print,time() - t1))
        # t1 = time()
    
    # Check if the loss and the parameters are not decreasing more than a tolerance from the previous epoch
    if epoch > 2*train_parameters_epoch and np.abs(losses[epoch,-1] - losses[epoch-1,-1]) < tol*losses[0,-1] and np.abs(param_values[epoch,0] - param_values[epoch-1,0]) < tol*param_values[0,0]:   
        print('Loss is not decreasing anymore. Stopping training.')
        break

# Print computation time
print('\nTotal training CPU time: {} seconds'.format(time() - t0))

#%%
# Plottings
#############################

# Plot the loss history (absolute)
plt.semilogy(losses[:epoch,0], '.', label='PDE')
plt.semilogy(losses[:epoch,1], '.', label='Data')
plt.semilogy(losses[:epoch,2], '.', label='IC')
plt.semilogy(losses[:epoch,3], '.', label='BC')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Unweighted Loss Function')
plt.legend()
plt.grid()
plt.show()

# Plot the loss history (weighted)
plt.semilogy(losses[:epoch,4], '.', label='PDE')
plt.semilogy(losses[:epoch,5], '.', label='Data')
plt.semilogy(losses[:epoch,6], '.', label='IC')
plt.semilogy(losses[:epoch,7], '.', label='BC')
plt.semilogy(losses[:epoch,8], '.', label='Total')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Weighted Loss Function')
plt.legend()
plt.grid()
plt.show()


# Plot the solutions
sol = model([t_data, x_data]).numpy().reshape(nt, nx)

# Plot solutions in space
for i in range(0, nt, 1):
    plt.plot(x_grid, sol[i, :],'k')#, label='c')
    plt.plot(x_grid, c_data_2d[i, :],'*r')#, label='c_data')
plt.xlabel('x')
plt.ylabel('C')
plt.title('Concentration vs space')
plt.legend()
plt.grid()
plt.show()

# Plot solutions in time
for i in range(nx):
    plt.plot(t_grid, sol[:,i],'k')#, label='c')
    plt.plot(t_grid, c_data_2d[:,i],'*r')#, label='c_data')
plt.xlabel('t')
plt.ylabel('C')
plt.title('Concentration vs time')
plt.legend()
plt.grid()
plt.show()

# Plot the parameters over time
for i in range(nparam):
    plt.plot(param_values[:epoch,i], label='p'+str(i))
    # plot a line representing the true parameter value
    plt.plot(np.ones(epoch)*params0[i]*randp[0], '--k', label='true value')
plt.xlabel('Epoch')
plt.ylabel('Parameter value')
plt.title('Parameters over time')
plt.legend()
plt.grid()
plt.show()


# Plot the parameter gradients over time
plt.semilogy(abs(param_grads[:epoch,0]), '*', label='d gradient')
# plt.semilogy(abs(param_hessian[:epoch,0]), '*', label='d Hessian')
plt.xlabel('Epoch')
plt.ylabel('Parameter gradients')
plt.title('Parameter gradients over time')
plt.legend()
plt.grid()
plt.show()


# %%
