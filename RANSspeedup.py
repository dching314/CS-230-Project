import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tecplot
import os
tecplot.session.acquire_license()

endStep = 20
stepGap = 2
Nsteps = int(endStep/stepGap)
Nvar = 15

dir_loc = 'E:\\RANS speedup\\CS230_bump'

# Training files
train_files = [dir_loc + '\\bump_H25_W50\\H25_W50-1',
               dir_loc + '\\bump_H15_W90\\H15_W90-1']

# target files
train_target_files = [dir_loc + '\\bump_H25_W50\\H25_W50-1-00305',
                      dir_loc + '\\bump_H15_W90\\H15_W90-1-00127']

# Test file
test_file = dir_loc + '\\bump_H15_W50\\H15_W50-1'
test_target_file = dir_loc + '\\bump_H15_W50\\H15_W50-1-00766'

# Initialize data arrays
print('Loading training/validation data...')
x = np.empty((Nvar*Nsteps,0))
y = np.empty((6,0))
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
        tmp = TPdataset.variable('Z Velocity').values(0)
        Vz = np.asarray(tmp[:])
        tmp = TPdataset.variable('Pressure').values(0)
        P = np.asarray(tmp[:])
        tmp = TPdataset.variable('Turbulent Kinetic Energy').values(0)
        k = np.asarray(tmp[:])
        tmp = TPdataset.variable('Turbulent Dissipation Rate').values(0)
        eps = np.asarray(tmp[:])
        tmp = TPdataset.variable('dX-Velocity/dx').values(0)
        dudx = np.asarray(tmp[:])
        tmp = TPdataset.variable('dX-Velocity/dy').values(0)
        dudy = np.asarray(tmp[:])
        tmp = TPdataset.variable('dX-Velocity/dz').values(0)
        dudz = np.asarray(tmp[:])
        tmp = TPdataset.variable('dY-Velocity/dx').values(0)
        dvdx = np.asarray(tmp[:])
        tmp = TPdataset.variable('dY-Velocity/dy').values(0)
        dvdy = np.asarray(tmp[:])
        tmp = TPdataset.variable('dY-Velocity/dz').values(0)
        dvdz = np.asarray(tmp[:])
        tmp = TPdataset.variable('dZ-Velocity/dx').values(0)
        dwdx = np.asarray(tmp[:])
        tmp = TPdataset.variable('dZ-Velocity/dy').values(0)
        dwdy = np.asarray(tmp[:])
        tmp = TPdataset.variable('dZ-Velocity/dz').values(0)
        dwdz = np.asarray(tmp[:])
#         tmp = TPdataset.variable('dkdx').values(0)
#         dkdx = np.asarray(tmp[:])
#         tmp = TPdataset.variable('dkdy').values(0)
#         dkdy = np.asarray(tmp[:])
#         tmp = TPdataset.variable('dkdz').values(0)
#         dkdy = np.asarray(tmp[:])

        if j == 0:
            time_step = np.empty((0,len(Vx)))
            
        tmp = np.stack((Vx,Vy,Vz,P,k,eps,dudx,dudy,dudx, dvdx,dvdy,dvdz, dwdx,dwdy,dwdz))
        time_step = np.append(time_step,tmp,axis=0)
        print('  Done loading: ' ,data_filename)
        
    print('Loaded files ' + str(i+1) + ' of ' + str(len(train_files)) + ', Grid size: ' + str(len(Vx)))
    x = np.append(x,time_step,axis=1)
    
    # Load tecplot target data
    case_filename = [train_files[i] + '.cas']
    data_filename = [train_target_files[i] + '.dat']
    TPdataset = tecplot.data.load_fluent(case_filenames=case_filename, data_filenames=data_filename, append=False) 
#     tecplot.data.operate.execute_equation('{dudx} = ddx({X Velocity})')
#     tecplot.data.operate.execute_equation('{dudy} = ddy({X Velocity})')
#     tecplot.data.operate.execute_equation('{dvdx} = ddx({Y Velocity})')
#     tecplot.data.operate.execute_equation('{dvdy} = ddy({Y Velocity})')

    # Place data into numpy arrays
    tmp = TPdataset.variable('X Velocity').values(0)
    Vx = np.asarray(tmp[:])
    tmp = TPdataset.variable('Y Velocity').values(0)
    Vy = np.asarray(tmp[:])
    tmp = TPdataset.variable('Z Velocity').values(0)
    Vz = np.asarray(tmp[:])
    tmp = TPdataset.variable('Pressure').values(0)
    P = np.asarray(tmp[:])
    tmp = TPdataset.variable('Turbulent Kinetic Energy').values(0)
    k = np.asarray(tmp[:])
    tmp = TPdataset.variable('Turbulent Dissipation Rate').values(0)
    eps = np.asarray(tmp[:])
    
    y = np.append(y,np.stack((Vx,Vy,Vz,P,k,eps)),axis=1)

m = x.shape[1]

# Initialize data arrays
print('Loading test data...')
x_test = np.empty((Nvar*Nsteps,0))
y_test = np.empty((6,0))
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
    tmp = TPdataset.variable('Z Velocity').values(0)
    Vz = np.asarray(tmp[:])
    tmp = TPdataset.variable('Pressure').values(0)
    P = np.asarray(tmp[:])
    tmp = TPdataset.variable('Turbulent Kinetic Energy').values(0)
    k = np.asarray(tmp[:])
    tmp = TPdataset.variable('Turbulent Dissipation Rate').values(0)
    eps = np.asarray(tmp[:])
    tmp = TPdataset.variable('dX-Velocity/dx').values(0)
    dudx = np.asarray(tmp[:])
    tmp = TPdataset.variable('dX-Velocity/dy').values(0)
    dudy = np.asarray(tmp[:])
    tmp = TPdataset.variable('dX-Velocity/dz').values(0)
    dudz = np.asarray(tmp[:])
    tmp = TPdataset.variable('dY-Velocity/dx').values(0)
    dvdx = np.asarray(tmp[:])
    tmp = TPdataset.variable('dY-Velocity/dy').values(0)
    dvdy = np.asarray(tmp[:])
    tmp = TPdataset.variable('dY-Velocity/dz').values(0)
    dvdz = np.asarray(tmp[:])
    tmp = TPdataset.variable('dZ-Velocity/dx').values(0)
    dwdx = np.asarray(tmp[:])
    tmp = TPdataset.variable('dZ-Velocity/dy').values(0)
    dwdy = np.asarray(tmp[:])
    tmp = TPdataset.variable('dZ-Velocity/dz').values(0)
    dwdz = np.asarray(tmp[:])
#     tmp = TPdataset.variable('dkdx').values(0)
#     dkdx = np.asarray(tmp[:])
#     tmp = TPdataset.variable('dkdy').values(0)
#     dkdy = np.asarray(tmp[:])
#     tmp = TPdataset.variable('dkdz').values(0)
#     dkdy = np.asarray(tmp[:])

    if j == 0:
        x_test = np.empty((0,len(Vx)))

    tmp = np.stack((Vx,Vy,Vz,P,k,eps,dudx,dudy,dudx, dvdx,dvdy,dvdz, dwdx,dwdy,dwdz))
    x_test = np.append(x_test,tmp,axis=0)
    print('  Done loading: ' ,data_filename)
    
case_filename = [test_file + '.cas']
data_filename = [test_target_file + '.dat']
TPdataset = tecplot.data.load_fluent(case_filenames=case_filename, data_filenames=data_filename, append=False) 


# Place data into numpy arrays
tmp = TPdataset.variable('X Velocity').values(0)
Vx = np.asarray(tmp[:])
tmp = TPdataset.variable('Y Velocity').values(0)
Vy = np.asarray(tmp[:])
tmp = TPdataset.variable('Z Velocity').values(0)
Vz = np.asarray(tmp[:])
tmp = TPdataset.variable('Pressure').values(0)
P = np.asarray(tmp[:])
tmp = TPdataset.variable('Turbulent Kinetic Energy').values(0)
k = np.asarray(tmp[:])
tmp = TPdataset.variable('Turbulent Dissipation Rate').values(0)
eps = np.asarray(tmp[:])

y_test = np.stack((Vx,Vy,Vz,P,k,eps))
    
# Move coords to cell center
tecplot.data.operate.execute_equation('{X1} = {X}', value_location=tecplot.constant.ValueLocation.CellCentered)
tecplot.data.operate.execute_equation('{Y1} = {Y}', value_location=tecplot.constant.ValueLocation.CellCentered)
tecplot.data.operate.execute_equation('{Z1} = {Z}', value_location=tecplot.constant.ValueLocation.CellCentered)
    
tmp = TPdataset.variable('X1').values(0)
X = np.asarray(tmp[:])
tmp = TPdataset.variable('Y1').values(0)
Y = np.asarray(tmp[:])
tmp = TPdataset.variable('Z1').values(0)
Z = np.asarray(tmp[:])

# Shuffle arrays
np.random.seed(2)
shuffle = np.random.permutation(m)
x = x[:,shuffle]
y = y[:,shuffle]

# Normalize data from result
# dataMean = np.zeros(Nvar)
# dataMean[0:6] = np.mean(y,axis=1)
dataMean = np.mean(x[Nsteps*Nvar-Nvar:,:], axis=1)
# dataStd = np.zeros(Nvar)
# dataStd[0:6] = np.std(y,axis=1)
dataStd = np.std(x[Nsteps*Nvar-Nvar:,:], axis=1)
yMean = np.mean(y,axis=1)
yStd = np.std(y,axis=1)

#print("Average values: ", normConst)
x_in = np.zeros(x.shape)
y_in = np.zeros(y.shape)
m = len(x[0,:])


for i in range(6):
    y_in[i,:] = (y[i,:] - yMean[(i)]) / yStd[(i)]
    y_test[i,:] = (y_test[i,:] - yMean[(i)]) / yStd[(i)]
    
for i in range(Nsteps*Nvar):
    x_in[i,:] = (x[i,:] - dataMean[(i%Nvar)]) / dataStd[(i%Nvar)]
    x_test[i,:] = (x_test[i,:] - dataMean[(i%Nvar)]) / dataStd[(i%Nvar)]


# Split into training, test, validation
frac_train = 0.98
frac_val = 1-frac_train

x_test = np.transpose(x_test)

x_train = np.transpose( x_in[:,0:int(m*frac_train)] )
x_val = np.transpose( x_in[:,int(m*frac_train):] )

y_train = np.transpose( y_in[:,0:int(m*frac_train)] )
y_val = np.transpose( y_in[:,int(m*frac_train):] )



def runModel(starter_learning_rate, decay_rate, decay_steps, epochs, batch_size, max_grad_norm, L, weightFile):
    tf.reset_default_graph()

    Nlayers = len(L)+1
    tf.set_random_seed(2)

    # declare the training data placeholders
    x = tf.placeholder(tf.float32, [None, Nvar*Nsteps])
    y = tf.placeholder(tf.float32, [None, 6])

#     T_1 = tf.placeholder(tf.float32, [None, 3, 3, 10])

    layers = {}
    layers['A0'] = x
    for i in range(Nlayers-1):
        layers['A' + str(i+1)] = tf.contrib.layers.fully_connected(layers['A' + str(i)], L[i], activation_fn=tf.nn.relu)

    yout = tf.contrib.layers.fully_connected(layers['A' + str(Nlayers-1)], 6, activation_fn=None)

#     # Reshape to tensorflow for multiplication by tensor
#     yLL = tf.expand_dims(tf.expand_dims(yLL, 1), 1)

#     # now calculate the TBNN output
#     yout = tf.reduce_sum(tf.multiply(yLL, T_1), axis=3)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    
    if weightFile == '':
        # Define error
        loss = tf.reduce_mean( tf.square( tf.subtract(yout, y)) )

        # add an optimiser
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.natural_exp_decay(starter_learning_rate, global_step, decay_steps, decay_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        grads_and_vars = optimizer.compute_gradients(loss)
        list_of_grads = [gv[0] for gv in grads_and_vars]

        grad_norm_before = tf.global_norm(list_of_grads)

        # clip the gradient
        list_of_grads, _ = tf.clip_by_global_norm(list_of_grads, max_grad_norm)

        # find the global norm of the gradients
        grad_norm = tf.global_norm(list_of_grads)

        #finally, reconstruct the original list and apply the gradients
        grads_and_vars_2 = [(list_of_grads[i], gv[1]) for i, gv in enumerate(grads_and_vars)]
        train_op = optimizer.apply_gradients(grads_and_vars_2) 


        # finally setup the initialization operator
        init_op = tf.global_variables_initializer()    


        # start the session
        m = len(x_train[:,0])
        total_batch = int(m / batch_size)

        # For plotting
        loss_train = np.empty(0)
        loss_test = np.empty(0)
        gradNorm_train = np.empty(0)
        gradNormClipped_train = np.empty(0)
        testLossSpace = 1   # Epochs between test

        with tf.Session() as sess:
           # initialise the variables
           sess.run(init_op)
           total_batch = int(m / batch_size)
           for epoch in range(epochs):
                avg_cost = 0
                avg_gradNormBefore = 0
                avg_gradNorm = 0

                # Shuffle data
                mask = np.random.permutation(m)
                shuffledTrain_x = x_train[mask]
                shuffledTrain_y = y_train[mask]
#                 shuffledTrain_T = T_train[mask]

                # Cut into batches
                for i in range(total_batch):
                    batch_x = shuffledTrain_x[i*batch_size:(i+1)*batch_size,:]
                    batch_y = shuffledTrain_y[i*batch_size:(i+1)*batch_size,:]           
#                     batch_T = shuffledTrain_T[i*batch_size:(i+1)*batch_size,:,:,:]

                    # Run
                    _, c, gradNormBefore, gradNorm = sess.run([train_op, loss, grad_norm_before, grad_norm], 
                                 feed_dict={x: batch_x, y: batch_y})
                    avg_cost += c / total_batch
                    avg_gradNormBefore += gradNormBefore / total_batch
                    avg_gradNorm += gradNorm / total_batch


                # Forward propogate validation data
                avg_cost_test, y_predicted = sess.run([loss, yout], feed_dict={x: x_val, y: y_val})

                # Save losses and gradients
                loss_train = np.append(loss_train, avg_cost)
                loss_test = np.append(loss_test, avg_cost_test)
                gradNorm_train = np.append(gradNorm_train, avg_gradNormBefore)
                gradNormClipped_train = np.append(gradNormClipped_train, avg_gradNorm)

                # Save model
                if epoch%20 == 19 and epoch > 0:
                    name = 'model'
                    for li in L:
                        name = name + '_' + (str(li))
                    name = name + '-' + str("{0:.6f}".format(starter_learning_rate)) + '-' + str("{0:.6f}".format(decay_rate)) + '-' + str(decay_steps) + '-' + str(batch_size) + '-' + str("{0:.3f}".format(max_grad_norm))  
                    name = name + '/epoch[' + str(epoch) + '] Loss' + str("{0:.6f}".format(avg_cost_test))
                    print(name)
                    save_path = saver.save(sess, "./NNmodelSave/" + name + ".ckpt")

                print("Epoch:", (epoch + 1), "cost =", "{a:.8f},   {b:.8f},   {c:.8f},   {d:.8f}"
                                  .format(a=avg_cost, b=avg_cost_test, c=avg_gradNormBefore, d=avg_gradNorm))

        plt.cla()
        plt.figure(1)
        plt.plot(np.arange(epochs), loss_train, np.arange(epochs), loss_test)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train','Test'])
        plt.show()


        f2 = plt.figure(2)
        plt.plot(np.arange(epochs), gradNorm_train, np.arange(epochs), gradNormClipped_train)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Before','After'])
        plt.show()
            
        return loss_train, loss_test, gradNorm_train, gradNormClipped_train
    
    else:
        with tf.Session() as sess:
          # Restore variables from disk.
            saver.restore(sess, "./NNmodelSave/" + weightFile + ".ckpt")

            # Forward propogate test data
            y_predicted = sess.run([yout], feed_dict={x: x_test})
        return y_predicted[0]
    
# Python optimisation variables
starter_learning_rate = 0.002
decay_rate = 0.0001
decay_steps = 4000
epochs = 100
batch_size = 500
max_grad_norm = .5

# Specify layer sizes
L = [40, 40, 40, 40, 40, 40] 

runModel(starter_learning_rate, decay_rate, decay_steps, epochs, batch_size, max_grad_norm, L,'')



# Run on test data
y_predict = runModel(starter_learning_rate, decay_rate, decay_steps, epochs, batch_size, max_grad_norm, L,
                     'model_40_40_40_40_40_40-0.002000-0.000100-4000-500-0.500/epoch[99] Loss0.004085')
Vx = y_predict[:,0] * dataStd[0] + dataMean[0]
Vy = y_predict[:,1] * dataStd[1] + dataMean[1]
Vz = y_predict[:,2] * dataStd[2] + dataMean[2]
P = y_predict[:,3] * dataStd[3] + dataMean[3]
k = y_predict[:,4] * dataStd[4] + dataMean[4]
eps = y_predict[:,5] * dataStd[5] + dataMean[5]

print(y_predict.shape)




def Write_Values(file, variable):
    file.write("(")
    
    for x in variable:
        file.write("{:.5e}\n".format(x))
        
    file.write(")\n")
    
def Write_Fluent_Interp_File(filename, names, x, y, z, list_uds):
    num_uds = len(list_uds)
    
    print("Writing interpolation file for {} case...".format(filename))
    
    N = x.size
    assert N == y.size and N == z.size, "Inconsistent sizes of variables"
    
    with open(filename, "w") as file_new:
    
#     file_new = open(filename, "w")
    
        file_new.write("3\n")
        file_new.write("3\n")
        file_new.write("{}\n".format(N))
        file_new.write("{}\n".format(num_uds))
        for i in range(num_uds):
            file_new.write("{}\n".format(names[i]))

        Write_Values(file_new, x)
        Write_Values(file_new, y)
        Write_Values(file_new, z)

        for i in range(num_uds): 
            uds = list_uds[i]
            assert uds.size == N, "UDS-{} has an inconsistent number of elements".format(i)
            Write_Values(file_new, uds)
    
    file_new.close()
    
    print("File written successfully.")
    
names = ['x-velocity','y-velocity','z-velocity','pressure','k','epsilon']
Write_Fluent_Interp_File('./Noninvariant-predicted.ip', names, X, Y, Z, (Vx, Vy, Vz, P, k, eps))
