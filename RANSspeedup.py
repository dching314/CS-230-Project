import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tecplot
import os
tecplot.session.acquire_license()




endStep = 50
stepGap = 2
Nsteps = int(endStep/stepGap)

# Training files
train_files = ['C:\\Users\\Eaton group\\Desktop\\CS230\\Project\\naca1206_aoa2\\Soln\\naca1206-2',
               'C:\\Users\\Eaton group\\Desktop\\CS230\\Project\\naca2412_aoa8\\Soln\\naca2412-1',
               'C:\\Users\\Eaton group\\Desktop\\CS230\\Project\\naca4424_aoa10\\Soln\\naca4424-4',
               'C:\\Users\\Eaton group\\Desktop\\CS230\\Project\\naca6412_aoa3\\Soln\\naca6412-1']

# target files
train_target_files = ['C:\\Users\\Eaton group\\Desktop\\CS230\\Project\\naca1206_aoa2\\Soln\\naca1206-2-00490',
               'C:\\Users\\Eaton group\\Desktop\\CS230\\Project\\naca2412_aoa8\\Soln\\naca2412-1-00695',
               'C:\\Users\\Eaton group\\Desktop\\CS230\\Project\\naca4424_aoa10\\Soln\\naca4424-4-00670',
               'C:\\Users\\Eaton group\\Desktop\\CS230\\Project\\naca6412_aoa3\\Soln\\naca6412-1-00703']

# Test file
test_file = 'C:\\Users\\Eaton group\\Desktop\\CS230\\Project\\naca0012_aoa5\\Soln\\naca0012-1'
test_target_file = 'C:\\Users\\Eaton group\\Desktop\\CS230\\Project\\naca0012_aoa5\\Soln\\naca0012-1-00677'

# Initialize data arrays
print('Loading training/validation data...')
x = np.empty((5*Nsteps,0))
y = np.empty((5,0))
for i in range(len(train_files)):
    Vx=Vy=P=k=eps= np.array([])
    for j in range(Nsteps):
        
        # Load tecplot training data
        case_filename = [train_files[i] + '.cas']
        data_filename = [train_files[i] + '-' + str(j*stepGap).zfill(5) + '.dat']
        TPdataset = tecplot.data.load_fluent(case_filenames=case_filename, data_filenames=data_filename, append=False) 

        # Place data into numpy arrays
        tmp = TPdataset.variable('X Velocity').values(0)
        Vx = np.asarray(tmp[:])
        tmp = TPdataset.variable('Y Velocity').values(0)
        Vy = np.asarray(tmp[:])
        tmp = TPdataset.variable('Pressure').values(0)
        P = np.asarray(tmp[:])
        tmp = TPdataset.variable('Turbulent Kinetic Energy').values(0)
        k = np.asarray(tmp[:])
        tmp = TPdataset.variable('Turbulent Dissipation Rate').values(0)
        eps = np.asarray(tmp[:])

        if j == 0:
            time_step = np.empty((0,len(Vx)))
            
        tmp = np.stack((Vx,Vy,P,k,eps))
        time_step = np.append(time_step,tmp,axis=0)
        
    print('Loaded files ' + str(i+1) + ' of ' + str(len(train_files)) + ', Grid size: ' + str(len(Vx)))
    x = np.append(x,time_step,axis=1)
    
    # Load tecplot target data
    case_filename = [train_files[i] + '.cas']
    data_filename = [train_target_files[i] + '.dat']
    TPdataset = tecplot.data.load_fluent(case_filenames=case_filename, data_filenames=data_filename, append=False) 

    # Place data into numpy arrays
    tmp = TPdataset.variable('X Velocity').values(0)
    Vx = np.asarray(tmp[:])
    tmp = TPdataset.variable('Y Velocity').values(0)
    Vy = np.asarray(tmp[:])
    tmp = TPdataset.variable('Pressure').values(0)
    P = np.asarray(tmp[:])
    tmp = TPdataset.variable('Turbulent Kinetic Energy').values(0)
    k = np.asarray(tmp[:])
    tmp = TPdataset.variable('Turbulent Dissipation Rate').values(0)
    eps = np.asarray(tmp[:])
    
    y = np.append(y,np.stack((Vx,Vy,P,k,eps)),axis=1)

m = x.shape[1]

# Initialize data arrays
print('Loading test data...')
x_test = np.empty((5*Nsteps,0))
y_test = np.empty((5,0))
Vx=Vy=P=k=eps= np.array([])
for j in range(Nsteps):

    # Load tecplot test data
    case_filename = [test_file + '.cas']
    data_filename = [test_file + '-' + str(j*stepGap).zfill(5) + '.dat']
    TPdataset = tecplot.data.load_fluent(case_filenames=case_filename, data_filenames=data_filename, append=False) 

    # Place data into numpy arrays
    tmp = TPdataset.variable('X Velocity').values(0)
    Vx = np.asarray(tmp[:])
    tmp = TPdataset.variable('Y Velocity').values(0)
    Vy = np.asarray(tmp[:])
    tmp = TPdataset.variable('Pressure').values(0)
    P = np.asarray(tmp[:])
    tmp = TPdataset.variable('Turbulent Kinetic Energy').values(0)
    k = np.asarray(tmp[:])
    tmp = TPdataset.variable('Turbulent Dissipation Rate').values(0)
    eps = np.asarray(tmp[:])

    if j == 0:
        x_test = np.empty((0,len(Vx)))

    tmp = np.stack((Vx,Vy,P,k,eps))
    x_test = np.append(x_test,tmp,axis=0)
    
case_filename = [test_file + '.cas']
data_filename = [test_target_file + '.dat']
TPdataset = tecplot.data.load_fluent(case_filenames=case_filename, data_filenames=data_filename, append=False) 

# Place data into numpy arrays
tmp = TPdataset.variable('X Velocity').values(0)
Vx = np.asarray(tmp[:])
tmp = TPdataset.variable('Y Velocity').values(0)
Vy = np.asarray(tmp[:])
tmp = TPdataset.variable('Pressure').values(0)
P = np.asarray(tmp[:])
tmp = TPdataset.variable('Turbulent Kinetic Energy').values(0)
k = np.asarray(tmp[:])
tmp = TPdataset.variable('Turbulent Dissipation Rate').values(0)
eps = np.asarray(tmp[:])

y_test = np.stack((Vx,Vy,P,k,eps))
    
# Move coords to cell center
tecplot.data.operate.execute_equation('{X1} = {X}', value_location=tecplot.constant.ValueLocation.CellCentered)
tecplot.data.operate.execute_equation('{Y1} = {Y}', value_location=tecplot.constant.ValueLocation.CellCentered)
    
tmp = TPdataset.variable('X1').values(0)
X = np.asarray(tmp[:])
tmp = TPdataset.variable('Y1').values(0)
Y = np.asarray(tmp[:])

# Shuffle arrays
np.random.seed(2)
shuffle = np.random.permutation(m)
x = x[:,shuffle]
y = y[:,shuffle]

# Normalize data from result
dataMean = np.mean(y,axis=1)
dataStd = np.std(y,axis=1)
#print("Average values: ", normConst)
x_in = np.zeros(x.shape)
y_in = np.zeros(y.shape)
m = len(x[0,:])

for i in range(5):
    y_in[i,:] = (y[i,:] - dataMean[(i)]) / dataStd[(i)]
    y_test[i,:] = (y_test[i,:] - dataMean[(i)]) / dataStd[(i)]
    
for i in range(Nsteps*5):
    x_in[i,:] = (x[i,:] - dataMean[(i%5)]) / dataStd[(i%5)]
    x_test[i,:] = (x_test[i,:] - dataMean[(i%5)]) / dataStd[(i%5)]


# Split into training, test, validation
frac_train = 0.99
frac_val = 1-frac_train

x_train = x_in[:,0:int(m*frac_train)]
x_val = x_in[:,int(m*frac_train):]

y_train = y_in[:,0:int(m*frac_train)]
y_val = y_in[:,int(m*frac_train):]






# Python optimisation variables
starter_learning_rate = 0.002
decay_rate = 0.01
decay_steps = 1000
epochs = 25
batch_size = 256

# Specify layer sizes
L = [Nsteps*5, 200, 150, 100, 5]      # fix last layer size to 5 and first to Nsteps*5
Nlayers = len(L)-1 

# declare the training data placeholders
x = tf.placeholder(tf.float32, [Nsteps*5, None])
# # now declare the output data placeholder - 5 digits
y = tf.placeholder(tf.float32, [5, None])

# Declare weights
parameters = {}
for i in range(0, Nlayers):
    parameters['W' + str(i+1)] = tf.Variable(tf.random_normal([L[i+1], L[i]], stddev=0.03), name='W'+str(i+1))
    parameters['b' + str(i+1)] = tf.Variable(tf.zeros([L[i+1], 1]), name='b'+str(i+1))
    
# first layer
parameters['Z1'] = tf.add(tf.matmul( parameters['W1'], x), parameters['b1'])
parameters['A1'] = tf.nn.relu(parameters['Z1'])
# rest of layers
for i in range(1, Nlayers-1):
    parameters['Z' + str(i+1)] = tf.add(tf.matmul( parameters['W' + str(i+1)], parameters['A' + str(i)]), parameters['b' + str(i+1)])
    parameters['A' + str(i+1)] = tf.nn.relu(parameters['Z' + str(i+1)])

# calculate the output of the last layer
y_out = tf.add(tf.matmul( parameters['W' + str(Nlayers)], parameters['A' + str(Nlayers-1)]), parameters['b' + str(Nlayers)])


# Define error
loss = tf.reduce_mean( tf.reduce_mean( tf.square( tf.subtract(y_out, y)) ))


# add an optimiser
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.natural_exp_decay(starter_learning_rate, global_step, decay_steps, decay_rate)
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

######################################################################
# start the session
sess = tf.Session()

# initialise the variables
sess.run(init_op)

# Couple parameters
m = len(x_train[0,:])
total_batch = int(m / batch_size)

# Start training
train_cost = []
val_cost = []
for epoch in range(epochs):
    avg_cost = 0
    for i in range(total_batch):
        batch_x = x_train[:,i*batch_size:(i+1)*batch_size]
        batch_y = y_train[:,i*batch_size:(i+1)*batch_size]

        _, c = sess.run([optimiser, loss],  feed_dict={x: batch_x, y: batch_y})
        avg_cost += c/total_batch
    train_cost = np.append(train_cost, avg_cost)
    l, _ = sess.run([loss, y_out],  feed_dict={x: x_val, y: y_val})
    val_cost = np.append(val_cost, l)
    print("Epoch", (epoch + 1), "of", (epochs), ", training cost =", "{:.3f}".format(train_cost[epoch]), ", val cost =", "{:.3f}".format(val_cost[epoch]))
    
    
    
    
    
    
    
    
    
# Plots the training history
fig = plt.figure()
plt.semilogy([i for i in range(1,epochs+1)],train_cost)
plt.semilogy([i for i in range(1,epochs+1)],val_cost)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()
fig.savefig('foo.png')












# Run on test data
loss, y_predict = sess.run([loss, y_out],  feed_dict={x: x_test, y: y_test})
Vx = y_predict[0,:] * dataStd[0] + dataMean[0]
Vy = y_predict[1,:] * dataStd[1] + dataMean[1]
P = y_predict[2,:] * dataStd[2] + dataMean[2]
k = y_predict[3,:] * dataStd[3] + dataMean[3]
eps = y_predict[4,:] * dataStd[4] + dataMean[4]

print('Loss on test set is: ', loss)













## Define all the helper functions that are used in the code
def Write_Values(file, variable):
    file.write("(")
    
    for x in variable:
        file.write("{:.5e}\n".format(x))
        
    file.write(")\n")

def Write_Fluent_Interp_File(geometry, x, y, list_uds):
    num_uds = len(list_uds)
    
    print("Writing interpolation file for {} case...".format(geometry))
    
    # the number of entries to write
    N = x.size
    assert N == y.size, "Inconsistent sizes of variables"
    
    # Variables to write. Number and name
#     names = ['uds-{}'.format(i) for i in range(num_uds)]
#     names = ['x-velocity','y-velocity','z-velocity','pressure', 'uds-0', 'uds-1', 'uds-2', 'uds-3', 'uds-4', 'uds-5']
    names = ['x-velocity','y-velocity','pressure','turb-kinetic-energy','turb-diss-rate']
    
    # Opening file
    new_filename = os.path.join("C:\\Users\\Eaton group\\Desktop\\CS230\\Project\\naca0012_aoa5\\Predicted", geometry + "_interpfile.ip")
    file_new = open(new_filename, "w")
    
    # Writing header
    file_new.write("3\n") # version. Must be 3
    file_new.write("2\n") # dimensionality. must be 3
    file_new.write("{}\n".format(N)) # number of points
    file_new.write("{}\n".format(num_uds))
    for i in range(num_uds):
        file_new.write("{}\n".format(names[i]))
    
    # Writing x,y,z values
    Write_Values(file_new, x)
    Write_Values(file_new, y)
    
    # Writing UDS themselves
    for i in range(num_uds): 
        uds = list_uds[i]
        assert uds.size == N, "UDS-{} has an inconsistent number of elements".format(i)
        Write_Values(file_new, uds)
    
    # Close file after we are done
    file_new.close()
    
    print("File written successfully.")
    
Write_Fluent_Interp_File('NACA0012_predicted', X, Y, (Vx, Vy, P, k, eps))