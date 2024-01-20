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

## Data and test case
testcase = "testcase0" # Testcase (choose the one you want to run)
coarsen_data = 1 # coarsening of the data (1 = no coarsening, >1 = coarsening skipping points)
data_perturbation = 0;#5e-1 # perturbation for the data

## Loss function weights (will be normalised afterwards)
pde_weight = 1.      # penalty for the PDE
data_weight = 1.     # penalty for the data fitting
ic_weight = 10.    # penalty for the initial condition
bc_weight = 10.     # penalty for the boundary condition

# NN training parameters
epochs = 5000          # number of epochs
learning_rate = 1e-2   # learning rate for the network weights
# Piecewise constant learning rate every Y epochs decayed by X
learning_rate_decay_factor = 0.9 # decay factor for the learning rate
learning_rate_step = 100
learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([learning_rate_step*(i+1) for i in range(int(epoch/learning_rate_step))],[learning_rate*learning_rate_decay_factor**i for i in range(int(epoch/learning_rate_step)+1)])

## NN architecture
num_hidden_layers = 8 # number of hidden layers (depth of the network)
num_neurons = 20      # max number of neurons per layer (width of the network)
def num_neurons_per_layer(depth): # number of neurons per layer (adapted to the depth of the network)
    return num_neurons    # constant number of neurons
    # return np.floor(num_neurons*(np.exp(-(depth-0.5)**2 * np.log(num_neurons/2.1)/((0.5)**2))))  # Gaussian distribution of neurons
activation = 'tanh' # 'sigmoid' or 'tanh'

## Parameters
train_parameters = True # train the parameters or not
param_perturbation = 0;#5e-1 # perturbation for the parameters
learning_rate_param = 1e-3 # correction factor for the learning rate of the parameters
train_parameters_epoch = 1000 # epoch after which train the parameters


#%%
# Load data
#############################

# Check if the folder exists
datafolder = "../data/"+testcase
if not os.path.exists(datafolder):
    # If it doesn't exist, use the folder name without the "../"
    datafolder = "data/"+testcase

# Load data as pandas dataframes
p = pd.read_csv(f'{datafolder}/p.csv', header=None)
x_grid = pd.read_csv(f'{datafolder}/x.csv', header=None)
t_grid = pd.read_csv(f'{datafolder}/t.csv', header=None)
c_data = pd.read_csv(f'{datafolder}/c.csv', header=None)
nt = np.shape(t_grid)[0]
nx = np.shape(x_grid)[0]
c_data_2d = np.reshape(c_data, (nt, nx))

# Flatten the coarsened array
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

# perturb data randomly
# c_data = c_data + data_perturbation * np.random.randn(c_data.size).reshape(c_data.shape) # additive noise
c_data = c_data * (1 + data_perturbation * np.random.randn(c_data.size).reshape(c_data.shape)) # multiplicative noise

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
randp =  (p * (param_perturbation * np.random.randn(p.size) + 1)).astype(np.float32)
beta0 = tf.Variable([randp[0]], trainable=False)
d = tf.Variable([randp[1]], trainable=train_parameters)
u = tf.Variable([randp[2]], trainable=False)
nparam = 1 # number of parameters to train

#%%
# Define the PINN model
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

@tf.function
def custom_loss(inputs, model, weights):

    # inputs
    xx, tt, cc = inputs

    # weights
    pde_weight, data_weight, ic_weight, bc_weight = weights
    
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
    
    # Compute the total loss divided by the total weights
    loss = (pde_weight*pde_loss + data_weight*data_fitting_loss + ic_weight*ic_fitting_loss + bc_weight*bc_fitting_loss) / (pde_weight + data_weight + ic_weight + bc_weight)

    return loss, pde_loss, data_fitting_loss, ic_fitting_loss, bc_fitting_loss


# Create the PINN model
model = pinn_model()
trainable = model.trainable_variables
if train_parameters:
    # trainable.append(beta0)
    # trainable.append(u)
    trainable.append(d)


#%%
# Train the NN
#############################

# Create the optimizer with a smaller learning rate
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate,epsilon=1e-08,amsgrad=True)

# some lists to save the losses and parameters
losses = np.zeros((epochs, 5))
param_values = np.zeros((epochs, nparam))

# Start loop and timer
stop = False
t0 = time()

# Epoch loop
for epoch in range(epochs):
    if not stop:
        # Compute the gradients
        with tf.GradientTape() as tape:
            # Call the tf decorated loss function
            loss = custom_loss(inputs, model, [pde_weight, data_weight*(epoch>train_parameters_epoch), ic_weight, bc_weight])
            gradients = tape.gradient(loss, trainable)
        
        # scale the gradients wrt to the parameters
        if (train_parameters):
            for i in range(-nparam, 0):
                gradients[i] *= learning_rate_param*(epoch>train_parameters_epoch)

        # Apply the gradients to update the weights
        optimizer.apply_gradients(zip(gradients, trainable))

        # Save the losses parameters and outputs to screen
        if epoch % 1 == 0:
            losses[epoch,:] = np.array(loss)
            param_values[epoch,0] = d.numpy()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss[0].numpy()}")
            print(f"pde_loss={loss[1].numpy()}, data_fitting_loss={loss[2].numpy()}, ic_fitting_loss={loss[3].numpy()}, bc_fitting_loss={loss[4].numpy()} ")                  
            print(f"beta0={beta0.numpy()}, d={d.numpy()}, u={u.numpy()} ")
        
        # # Check if the loss is not decreasing anymore
        # if len(losses) > 2 and (np.abs(losses[-1] - losses[-2]))/np.abs(losses[-2]) < 1e-8:
        #     stop = True

# Print computation time
print('\nComputation time: {} seconds'.format(time() - t0))

#%%
# Plottings
#############################

# Plot the loss history (relative)
plt.semilogy(losses[:,0]/losses[0,0], label='Total Loss')
plt.semilogy(losses[:,1]/losses[0,1], label='PDE Loss')
plt.semilogy(losses[:,2]/losses[0,2], label='Data Fitting Loss')
plt.semilogy(losses[:,3]/losses[0,3], label='IC Fitting Loss')
plt.semilogy(losses[:,4]/losses[0,4], label='BC Fitting Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Normalised Loss Function')
plt.legend()
plt.grid()
plt.show()

# Plot the loss history (absolute)
plt.semilogy(losses[:,0], label='Total Loss')
plt.semilogy(losses[:,1], label='PDE Loss')
plt.semilogy(losses[:,2], label='Data Fitting Loss')
plt.semilogy(losses[:,3], label='IC Fitting Loss')
plt.semilogy(losses[:,4], label='BC Fitting Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Absolute Loss Function')
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
plt.plot(param_values[:,0], label='d')
plt.xlabel('Epoch')
plt.ylabel('Parameter value')
plt.title('Parameters over time')
plt.legend()
plt.grid()
plt.show()


# %%
