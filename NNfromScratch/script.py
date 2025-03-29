# %%
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_gaussian_quantiles

# %%
samples = np.array([
    [1,2,3],
    [6,7,8],
    [7,8,9],
    [3,4,5],
    [4,5,6],
])

X = samples
targets = np.array([False, True, True, False, False])

# %%
uniques, indices = np.unique(targets, return_inverse = True)
print(f"Original array : {targets}")
print(f"Unique array : {uniques}")
print(f"Indices : {indices}")

# %%
n_samples = targets.shape[0]
n_classes = len(uniques)
y = np.zeros((n_samples, n_classes))
print(y)


# %%
print(np.arange(n_samples))
print(indices)

# %%
y[np.arange(n_samples), indices] = 1
print(targets, '\n')
print("one hot encoding targets:")
print(y)

# %%
# The shape of our dataset
print(X.shape)
n_features = X.shape[1]

print(f"Dataset size : {n_samples}")
print(f"Features size : {n_features}")

# %%
# the number of units in the hidden layer
n_hidden_units = 4

# %%
np.random.seed(10)

Wh = np.random.uniform(low=-0.5, high=0.5, size=(n_features, n_hidden_units))
bh = np.zeros((1, n_hidden_units))

# %%
print(Wh)

# %%
print(f"input shape: {X.shape}")
print(f"hidden weights shape : {Wh.shape}")
print(f"hidden biases shape: {bh.shape}")

# %%
# the weights of the first hidden unit
# reshape is used just to display the result in column format
print(Wh, '\n')
print("Weights of the first hidden unit: ")
print(Wh[:,0].reshape(3,1)) 
# this code snippet is for explaination only

# %%
h1 = np.dot(X, Wh) + bh
print(h1.shape)
print(h1)

# %%
# passing values thru relU
a1 = np.maximum(0,h1)
print("before  ReLU (h1) :")
print(h1, '\n')
print("After ReLU (a1): ")
print(a1) # a1 is the output of the hidden layer

# %%
np.random.seed(100)

Wo = np.random.uniform(low=-0.5, high=0.5, size = (n_hidden_units, n_classes))
bo = np.zeros((1,n_classes))


# %%
print(Wo)

# %%
print(f"Hidden layer output shape: {a1.shape}")
print(f"Output weights shape: {Wo.shape}")
print(f"Output biases shape: {bo.shape}")

# %%
h2 = np.dot(a1, Wo) + bo
print(h2.shape)
print(h2)

# %%
# softmax func
#first we will calculate the numerators
e_x = np.exp(h2)
print(e_x)

# %%
print(np.exp(10))
print(np.exp(100))
print(np.exp(1000))

# %%
np.max(h2)

# %%
print(h2, '\n')
print("Maximum value from each row: ")
print(np.max(h2,axis=1))

# %%
# this will show error
#as we are subtracting the 2 columns from the 5 columns
np.exp(h2 - np.max(h2, axis=1))

# %%
np.max(h2, axis=1, keepdims = True) #hence we make the row and column same

# %%
# we can now calculate
e_x = np.exp(h2 - np.max(h2, axis = 1, keepdims = True))
print(e_x)

# %%
#now we will calculate the softmax denominators
np.sum(e_x, axis=1, keepdims = True) 

# %%
y_hat = e_x / np.sum(e_x, axis=1, keepdims= True) #  according to softmax formula
y_hat

# %%
#calculate CCE loss (full version)
print(y)
print(y_hat)
  

# %%
#component wise multiply and summation in each row
np.sum(y * -np.log(y_hat), axis = 1)

# %%
y_hat_clipped = np.clip(y_hat, np.finfo(float).eps, 1 - np.finfo(float).eps) # clip function is used so that we can use it to to let log(0) to show error
print(y_hat_clipped)

# %%
neg_logs = np.sum(y * -np.log(y_hat_clipped), axis = 1)
neg_logs

# %%
cce_loss = np.mean(neg_logs)
print(f"The loss after this forward pass is : {cce_loss}") #this completes our forward pass

# %%
print("y")
print(y, '\n')
print("y-hat")
print(y_hat_clipped)

# %%
# we will normalize also so that the gradient is not affected by the size of the gradient.
dloss_dh2 = (y_hat - y) / n_samples
print(dloss_dh2)

# %%
dh2_dWo = a1
print(dh2_dWo)

# %%
# now we will multiply 
print(f"Wo: {Wo.shape}")

# %%
print(f"dh2_dwo: {dh2_dWo.shape}")
print(f"dloss_dh2: {dloss_dh2.shape}")

# %%
print(f'{dh2_dWo.T.shape} * {dloss_dh2.shape}') # we take the transpose

# %%
print(a1)

# %%
print(dloss_dh2)

# %%
dh2_dWo = a1.T
print(dh2_dWo)

# %%
#now we can calculate the derivative of Lcce wrt W0
#gradient of the output weight
dloss_dWo = np.dot(dh2_dWo, dloss_dh2)
print("The gradient for the outptut weights (Wo) :")
print(dloss_dWo)

# %%
# gradient of the output baises
dloss_dbo = np.sum(dloss_dh2, axis =0, keepdims= True)
print("The gradient for the output biases (bo):")
print(dloss_dbo)

# %%
 #now we will calculate the gradient of the hidden weights and biases
dh2_da1 = Wo.T
print(dh2_da1.shape)

# %%
print(dloss_dh2.shape)
print(Wo.T.shape)


# %%
print(Wo)

# %%
print("Weights between first hidden unit and each output unit:")
print(Wo[0])

# %%
print(dloss_dh2)

# %%
dloss_da1 = np.dot(dloss_dh2, dh2_da1)
print(dloss_da1.shape)
print(dloss_da1)

# %%
da1_dh1 = np.zeros(h1.shape, dtype = np.float32)
da1_dh1[h1 > 0] = 1
print(h1, '\n')
print(da1_dh1)

# %%
print(f"dloss_da1: {dloss_da1.shape}")
print(f"da1_dh1: {da1_dh1.shape}")

# %%
dloss_dh1 = da1_dh1 * dloss_da1
print(dloss_dh1)

# %%
dh1_dWo = X.T


# %%
dloss_dWh = np.dot(dh1_dWo, dloss_dh1)


# %%
print(f"Hidden weights: {Wh.shape}")
print(f"dh1_dWo: {dloss_dWh.shape}")

# %%
dloss_dbh = np.sum(dloss_dh1, axis=  0, keepdims = True)
dloss_dbh

# %%
#learning rate
lr = 0.01

#upadtes output weights and biases
new_Wo = Wo - lr*dloss_dWo
new_bo = bo - lr * dloss_dbo

#updates hidden weights and biases.
new_Wh = Wh - lr * dloss_dWh
new_bh = bh - lr * dloss_dbh

#This completes our backward pass

# %%
# To check if the loss minimized or not we again do a forward pass with thenew Values
h1 = np.dot(X, new_Wh) + new_bh
a1 = np.maximum(0, h1)
h2 = np.dot(a1, new_Wo) + new_bo

# Softmax
e_x = np.exp(h2 - np.max(h2, axis=1, keepdims=True))
y_hat = e_x / np.sum(e_x, axis=1, keepdims=True)
y_hat_clipped = np.clip(y_hat, np.finfo(float).eps, 1 - np.finfo(float).eps)

# Cross entropy
neg_logs = np.sum(y * -np.log(y_hat_clipped), axis=1)

new_cce_loss = np.mean(neg_logs)

print(f'New loss: {new_cce_loss}')
print(f'Previous loss: {cce_loss}')

# Hence the new loss decreased

# %%
# But this way is very lenthy confusing and not applicable for large scale
# therefore we will be using frameworks for faster implementation int any big projects. 


