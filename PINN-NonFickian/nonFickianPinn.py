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

#%%
# Testcase (choose the one you want to run)
datafolder = "../data/testcase1"
data_weight = 1000
ic_weight = 10000
learning_rate = 1e-4


# Load data
p = pd.read_csv(f'{datafolder}/p.csv', header=None)
x_data = pd.read_csv(f'{datafolder}/x.csv', header=None)
t_data = pd.read_csv(f'{datafolder}/t.csv', header=None)
c_data = pd.read_csv(f'{datafolder}/c.csv', header=None)
c1_data = pd.read_csv(f'{datafolder}/c1.csv', header=None)

nt = np.shape(t_data)[0]
nx = np.shape(x_data)[0]

X_data, T_data = np.meshgrid(x_data, t_data)
Xgrid = np.vstack([X_data.flatten(), T_data.flatten()]).T
Xgrid_data = Xgrid[:, 0]
Tgrid_data = Xgrid[:, 1]

p = np.array(p).squeeze().astype(np.float32)
x_train = Xgrid_data.astype(np.float32)
t_train = Tgrid_data.astype(np.float32)
c_train = c_data.astype(np.float32)
c1_train = c1_data.astype(np.float32)

# additional variables added to gradient tracking
beta0 = tf.Variable([p[0]], trainable=False)
d = tf.Variable([p[1]], trainable=False)
u = tf.Variable([p[2]], trainable=False)
beta1 = tf.Variable([p[3]], trainable=False)
lambda1 = tf.Variable([p[4]], trainable=False)

#%%
### Original ###
# def PINNModel():
#     x_input = keras.Input(shape=(1,))
#     t_input = keras.Input(shape=(1,))
#
#     combined = layers.concatenate([x_input, t_input])
#
#     # NN for theta
#     # Process x using dense layers with non-negative kernels
#     output_c = keras.Sequential([
#         layers.Dense(128,
#                      activation='relu',
#                      kernel_initializer='glorot_normal',
#                      # kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
#                      ),
#         layers.Dense(128,
#                      activation='relu',
#                      kernel_initializer='random_normal',
#                      # kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
#                      ),
#         layers.Dense(128,
#                      activation='relu',
#                      kernel_initializer='glorot_normal',
#                      # kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
#                      ),
#         layers.Dense(2,
#                      activation='relu',
#                      kernel_initializer='random_normal',
#                      # kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
#                      )
#     ])(combined)
#
#     return keras.Model(inputs=[x_input, t_input], outputs=output_c)

def pinn_model(num_hidden_layers=8, num_neurons_per_layer=20):
    x_input = keras.Input(shape=(1,))
    t_input = keras.Input(shape=(1,))

    combined = layers.concatenate([x_input, t_input])

    output_c = combined
    for _ in range(num_hidden_layers):
        output_c = tf.keras.layers.Dense(num_neurons_per_layer,
                                         activation='tanh',
                                         kernel_constraint=NonNeg(),
                                         kernel_initializer='glorot_normal',
                                         # kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
                                         )(output_c)

    return keras.Model(inputs=[x_input, t_input], outputs=output_c)


# @tf.function
def custom_loss(inputs, model):
    x, t, c_data, c1_data = inputs

    # Compute derivatives:
    tt = tf.Variable(t)
    xx = tf.Variable(x)
    with tf.GradientTape(persistent=True) as tape:
        output_model = model([xx, tt])
        c_model = output_model[:, 0]
        c_model = tf.expand_dims(c_model, -1)
        c1_model = output_model[:, 1]
        c1_model = tf.expand_dims(c1_model, -1)
        tape.watch(tt)
        tape.watch(xx)
        c_x = tape.gradient(c_model, xx)
    c_t = tape.gradient(c_model, tt)
    c1_t = tape.gradient(c1_model, tt)
    div_output = u * c_x - d * tape.gradient(c_x, xx)

    # Compute the components of loss function
    pde_loss_c = tf.reduce_mean(
        (tf.multiply(beta0, c_t) + div_output + tf.multiply(tf.multiply(beta1, lambda1), c1_model - c_model)) ** 2)
    pde_loss_c1 = tf.reduce_mean((c1_t - tf.multiply(lambda1, c1_model - c_model)) ** 2)
    pde_loss = pde_loss_c + pde_loss_c1
    data_c_fitting_loss = tf.reduce_mean((c_model - c_data) ** 2)
    data_c1_fitting_loss = tf.reduce_mean((c1_model - c1_data) ** 2)
    data_fitting_loss = data_c_fitting_loss + data_c1_fitting_loss
    ic_c1_fitting_loss = tf.reduce_mean(c1_model[0:nt] ** 2)
    loss = pde_loss + data_weight*data_fitting_loss + ic_weight*ic_c1_fitting_loss

    del tape

    return loss, pde_loss, data_fitting_loss, ic_c1_fitting_loss


# Create the PINN model
model = pinn_model()
trainable = model.trainable_variables
trainable.append(beta0)
trainable.append(beta1)
trainable.append(lambda1)
trainable.append(u)
trainable.append(d)

epochs = 100  # 1000
# # Compile the model
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
#               loss=lambda y_true, y_pred: custom_loss([x_train, t_train, theta_train], model)[1])

# Create the optimizer with a smaller learning rate
# learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([10, 100], [1e-1, 5e-2, 1e-2])  #OK
# learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([100, 300], [1e-2, 1e-3, 5e-4])
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

# Training loop
losses = []
pde_losses = []
data_fitting_losses = []
ic_c1_fitting_losses = []
beta0_values = []
beta1_values = []
lambda1_values = []
u_values = []
d_values = []
# sigma_values = []


# Convert data to tensor because tf.GradientTape() can only watch tensor and not numpy arrays
x_train = tf.expand_dims(tf.convert_to_tensor(x_train), -1)
t_train = tf.expand_dims(tf.convert_to_tensor(t_train), -1)
c_data_train = tf.convert_to_tensor(c_train)
c1_data_train = tf.convert_to_tensor(c1_train)
inputs = [x_train, t_train, c_data_train, c1_data_train]
stop = False

# Start timer
t0 = time()
for epoch in range(epochs):
    if not stop:
        print("# STARTING EPOCH", epoch + 1)

        with tf.GradientTape() as tape:
            loss, pde_loss, data_fitting_loss, ic_c1_fitting_loss = custom_loss(inputs, model)

        print("Computing gradients")
        gradients = tape.gradient(loss, trainable)
        print("Applying gradients")
        optimizer.apply_gradients(zip(gradients, trainable))
        print("Appending losses")
        losses.append(loss.numpy())
        pde_losses.append(pde_loss.numpy())
        data_fitting_losses.append(data_fitting_loss.numpy())
        ic_c1_fitting_losses.append(ic_c1_fitting_loss.numpy())
        beta0_values.append(beta0.numpy())
        beta1_values.append(beta1.numpy())
        lambda1_values.append(lambda1.numpy())
        u_values.append(u.numpy())
        d_values.append(d.numpy())

        current_lr = optimizer._decayed_lr(tf.float32).numpy()
        # beta0.assign_sub(gradients[-5] * current_lr)
        # beta1.assign_sub(gradients[-4] * current_lr)
        # lambda1.assign_sub(gradients[-3] * current_lr)
        # u.assign_sub(gradients[-2] * current_lr)
        # d.assign_sub(gradients[-1] * current_lr)

        if epoch % 1 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}")
            print(
                f"beta0={beta0.numpy()}, beta1={beta1.numpy()}, lambda1={lambda1.numpy()}, u={u.numpy()}, d={d.numpy()}")

        if len(losses) > 2 and (np.abs(losses[-1] - losses[-2]))/np.abs(losses[-2]) < 1e-8:
            stop = True

# Print computation time
print('\nComputation time: {} seconds'.format(time() - t0))

# Plot the loss history
plt.plot(losses, label='Total Loss')
plt.plot(pde_losses, label='PDE Loss')
plt.plot(data_fitting_losses, label='Data Fitting Loss')
plt.plot(ic_c1_fitting_losses, label='IC c1 Fitting Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Contributions')
plt.legend()
plt.grid()
plt.show()


solutions = model([x_train, t_train])
sol_c = solutions[:, 0]
sol_c1 = solutions[:, 1]

sol_c = tf.reshape(sol_c, [x_data.shape[0], t_data.shape[0]])
sol_c1 = tf.reshape(sol_c1, [x_data.shape[0], t_data.shape[0]])
c_data_ = tf.reshape(c_data, [x_data.shape[0], t_data.shape[0]])
c1_data_ = tf.reshape(c1_data, [x_data.shape[0], t_data.shape[0]])
# Plot solutions
for i in range(3):#sol_c.shape[1]):
    plt.plot(x_data, sol_c[:, i])#, label='c')
    plt.plot(x_data, c_data_[i, :])#, label='c_data')
plt.xlabel('x')
plt.ylabel('fun')
plt.title('Comparison')
plt.legend()
plt.grid()
plt.show()

for i in range(3):#sol_c.shape[1]):
    plt.plot(x_data, sol_c1[:, i])#, label='c1')
    plt.plot(x_data, c1_data_[i, :])#, label='c1_data')
plt.xlabel('x')
plt.ylabel('fun')
plt.title('Comparison')
plt.legend()
plt.grid()
plt.show()

# %%
