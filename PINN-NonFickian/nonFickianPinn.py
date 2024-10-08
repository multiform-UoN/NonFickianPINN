#%%
# import
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

seed = int(time())
np.random.seed(seed)

# Configure TensorFlow to use the specified number of threads
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
 
#%%
# USER INPUTS

testcase = "testcase2" # Testcase (choose the one you want to run)

pde_weight = 1.0      # penalty for the PDE
data_weight = 1.0     # penalty for the data fitting
ic_weight = 100.0     # penalty for the initial condition
bc_weight = 100.0     # penalty for the boundary condition

# learning_rate = 3e-4   # learning rate for the network weights
# learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([10, 100], [1e-1, 5e-2, 1e-2])  #OK
learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([100, 300], [1e-3, 1e-4, .5e-4])

epochs = 10000          # number of epochs
num_hidden_layers = 20 # number of hidden layers (depth of the network)
num_neurons = 50      # max number of neurons per layer (width of the network)
def num_neurons_per_layer(depth): # number of neurons per layer (adapted to the depth of the network)
    return num_neurons    # constant number of neurons
    # return np.floor(num_neurons*(np.exp(-(depth-0.5)**2 * np.log(num_neurons/2.1)/((0.5)**2))))  # Gaussian distribution of neurons
activation = 'tanh' # 'sigmoid' or 'tanh'

train_parameters = False # train the parameters or not
def learning_rate_parameters(learning_rate, epoch): # learning rate for parameters
    return learning_rate*0
param_perturbation = 5e-1 # perturbation for the parameters

# Check if the folder exists
datafolder = "../data/"+testcase
if not os.path.exists(datafolder):
    # If it doesn't exist, use the folder name without the "../"
    datafolder = "data/"+testcase

# Load data as pandas dataframes
p = pd.read_csv(f'{datafolder}/p.csv', header=None)
x_data = pd.read_csv(f'{datafolder}/x.csv', header=None)
t_data = pd.read_csv(f'{datafolder}/t.csv', header=None)
c_data = pd.read_csv(f'{datafolder}/c.csv', header=None)
c1_data = pd.read_csv(f'{datafolder}/c1.csv', header=None)

# mesh and time discretization
nt = np.shape(t_data)[0]
nx = np.shape(x_data)[0]
X_data, T_data = np.meshgrid(x_data, t_data)
Xgrid = np.vstack([X_data.flatten(), T_data.flatten()]).T
Xgrid_data = Xgrid[:, 0]
Tgrid_data = Xgrid[:, 1]

# convert data to numpy arrays and float32
p = np.array(p).squeeze().astype(np.float32)
x_train = Xgrid_data.astype(np.float32)
t_train = Tgrid_data.astype(np.float32)
c_train = c_data.astype(np.float32)
c1_train = c1_data.astype(np.float32)

# Convert data to tensor because tf.GradientTape() can only watch tensor and not numpy arrays
x_train = tf.expand_dims(tf.convert_to_tensor(x_train), -1)
t_train = tf.expand_dims(tf.convert_to_tensor(t_train), -1)
c_data_train = tf.convert_to_tensor(c_train)
c1_data_train = tf.convert_to_tensor(c1_train)

# tf differentiation variables
tt = tf.Variable(t_train)
xx = tf.Variable(x_train)

# model inputs
inputs = [xx, tt, c_data_train, c1_data_train]


# additional variables added to gradient tracking
randp =  (p * (param_perturbation * np.random.randn(p.size) + 1)).astype(np.float32)
beta0 = tf.Variable([randp[0]], trainable=train_parameters)
d = tf.Variable([randp[1]], trainable=train_parameters)
u = tf.Variable([randp[2]], trainable=train_parameters)
beta1 = tf.Variable([randp[3]], trainable=train_parameters)
lambda1 = tf.Variable([randp[4]], trainable=train_parameters)

#%%
# Define the PINN model
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
    
    # # output layer
    # output_c = tf.keras.layers.Dense(2,
    #                                 activation=None  # to check
    #                                 )(output_c)

    return keras.Model(inputs=[t_input, x_input], outputs=output_c)

@tf.function
def custom_loss(inputs, model):

    # inputs
    xx, tt, c_data, c1_data = inputs
    
    # Compute derivatives:
    with tf.GradientTape(persistent=True) as tape:
        # # # variables
        # tt = tf.Variable(t)
        # xx = tf.Variable(x)
        tape.watch(tt)
        tape.watch(xx)
        # output model
        output_model = model([tt, xx])
        c_model = output_model[:, 0]
        c_model = tf.expand_dims(c_model, -1)
        c1_model = output_model[:, 1]
        c1_model = tf.expand_dims(c1_model, -1)
        # derivatives and fluxes
        c_x = tape.gradient(c_model, xx)
        c_t = tape.gradient(c_model, tt)
        c1_t = tape.gradient(c1_model, tt)
        div_output = u * c_x - d * tape.gradient(c_x, xx)

    del tape

    # Compute the components of loss function
    pde_loss_c = tf.reduce_mean(
        (tf.multiply(beta0, c_t) + div_output + tf.multiply(tf.multiply(beta1, lambda1), c1_model - c_model)) ** 2)
    pde_loss_c1 = tf.reduce_mean((c1_t - tf.multiply(lambda1, c1_model - c_model)) ** 2)
    pde_loss = pde_loss_c + pde_loss_c1
    data_c_fitting_loss = tf.reduce_mean((c_model - c_data) ** 2)
    data_bc_fitting_loss = tf.reduce_mean((c_model[::nx] - c_data[::nx]) ** 2)
    data_c1_fitting_loss = tf.reduce_mean((c1_model - c1_data) ** 2)
    data_fitting_loss = data_c_fitting_loss + data_c1_fitting_loss
    ic_c1_fitting_loss = tf.reduce_mean((c1_model[0:nt] - c1_data[0:nt]) ** 2)
    ic_c_fitting_loss = tf.reduce_mean((c_model[0:nt] - c_data[0:nt])  ** 2)
    ic_fitting_loss = ic_c_fitting_loss + ic_c1_fitting_loss
    loss = pde_weight*pde_loss + data_weight*data_fitting_loss + ic_weight*ic_fitting_loss + bc_weight*data_bc_fitting_loss


    return loss, pde_loss, data_fitting_loss, ic_fitting_loss, data_bc_fitting_loss


# Create the PINN model
model = pinn_model()
trainable = model.trainable_variables
if train_parameters:
    trainable.append(beta0)
    trainable.append(beta1)
    trainable.append(lambda1)
    trainable.append(u)
    trainable.append(d)

# # Compile the model
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
#               loss=lambda y_true, y_pred: custom_loss([x_train, t_train, theta_train], model)[1])

# Create the optimizer with a smaller learning rate
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

# some lists to save the losses and parameters
losses = []
pde_losses = []
data_fitting_losses = []
ic_fitting_losses = []
data_bc_fitting_losses = []
beta0_values = []
beta1_values = []
lambda1_values = []
u_values = []
d_values = []

# Start loop and timer
stop = False
t0 = time()

# Epoch loop
for epoch in range(epochs):
    if not stop:
        # Compute the gradients
        with tf.GradientTape() as tape:
            # Call the tf decorated loss function
            loss, pde_loss, data_fitting_loss, ic_fitting_loss, data_bc_fitting_loss = custom_loss(inputs, model)
            gradients = tape.gradient(loss, trainable)
        
        # Apply the gradients to update the weights
        optimizer.apply_gradients(zip(gradients, trainable))

        # # Update parameters
        # current_lr = learning_rate_parameters(optimizer.learning_rate.numpy(), epoch)
        # # perhaps this needs to be applied only after a certain number of epochs
        # if train_parameters: 
        #     beta0.assign_sub(gradients[-5] * current_lr)
        #     beta1.assign_sub(gradients[-4] * current_lr)
        #     lambda1.assign_sub(gradients[-3] * current_lr)
        #     u.assign_sub(gradients[-2] * current_lr)
        #     d.assign_sub(gradients[-1] * current_lr)

        # Save the losses parameters and outputs to screen
        if epoch % 1 == 0:
            losses.append(loss.numpy())
            pde_losses.append(pde_loss.numpy())
            data_fitting_losses.append(data_fitting_loss.numpy())
            data_bc_fitting_losses.append(data_bc_fitting_loss.numpy())
            ic_fitting_losses.append(ic_fitting_loss.numpy())
            beta0_values.append(beta0.numpy())
            beta1_values.append(beta1.numpy())
            lambda1_values.append(lambda1.numpy())
            u_values.append(u.numpy())
            d_values.append(d.numpy())
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}")
            print(f"beta0={beta0.numpy()}, d={d.numpy()}, u={u.numpy()}, beta1={beta1.numpy()}, lambda1={lambda1.numpy()} ")

        # Check if the loss is not decreasing anymore
        if len(losses) > 2 and (np.abs(losses[-1] - losses[-2]))/np.abs(losses[-2]) < 1e-8:
            stop = True

# Print computation time
print('\nComputation time: {} seconds'.format(time() - t0))

#%%
# Plot the loss history (relative)
plt.semilogy(losses/losses[0], label='Total Loss')
plt.semilogy(pde_losses/pde_losses[0], label='PDE Loss')
plt.semilogy(data_fitting_losses/data_fitting_losses[0], label='Data Fitting Loss')
plt.semilogy(ic_fitting_losses/ic_fitting_losses[0], label='IC Fitting Loss')
plt.semilogy(data_bc_fitting_losses/data_bc_fitting_losses[0], label='BC Fitting Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.title('Loss Contributions')
plt.legend(loc='best')
plt.grid()
plt.show()

# Plot the loss history (absolute)
plt.semilogy(losses, label='Total Loss')
plt.semilogy(pde_losses, label='PDE Loss')
plt.semilogy(data_fitting_losses, label='Data Fitting Loss')
plt.semilogy(ic_fitting_losses, label='IC Fitting Loss')
plt.semilogy(data_bc_fitting_losses, label='BC Fitting Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.title('Loss Contributions')
plt.legend(loc='best')
plt.grid()
plt.show()


#%%
# Plot the solutions
solutions = model([t_train, x_train])
sol_c = solutions[:, 0]
sol_c1 = solutions[:, 1]

sol_c_ = tf.reshape(sol_c, [t_data.shape[0], x_data.shape[0]])
sol_c1_ = tf.reshape(sol_c1, [t_data.shape[0], x_data.shape[0]])
c_data_ = tf.reshape(c_data, [t_data.shape[0], x_data.shape[0]])
c1_data_ = tf.reshape(c1_data, [t_data.shape[0], x_data.shape[0]])
# Plot solutions
for i in range(sol_c_.shape[0]):
    plt.plot(x_data, sol_c_[i, :],'k')#, label='c')
    plt.plot(x_data, c_data_[i, :],'*r')#, label='c_data')
plt.xlabel('x')
plt.ylabel('fun')
# plt.title('Comparison')
plt.legend(loc='best')
plt.grid()
plt.show()

for i in range(sol_c_.shape[0]):
    plt.plot(x_data, sol_c1_[i, :],'k')#, label='c1')
    plt.plot(x_data, c1_data_[i, :],'*r')#, label='c1_data')
plt.xlabel('x')
plt.ylabel('fun')
# plt.title('Comparison')
plt.legend(loc='best')
plt.grid()
plt.show()

# %%
