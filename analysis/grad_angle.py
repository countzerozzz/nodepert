import npimports
import importlib
importlib.reload(npimports)
from npimports import *

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def get_grad_angles(params, randkey, optimstate):
    xl, yl = next(iter(data.get_data_batches(batchsize=10000, split=data.trainsplit)))
    _, gradtrue, optimstate = optim.sgdupdate(xl, yl, params, randkey, optimstate)
    tmp={}

    for ii in range(len(layer_sizes)-1):
        tmp['w'+str(ii)+'np']=[]
        tmp['b'+str(ii)+'np']=[]
        tmp['w'+str(ii)+'sgd']=[]
        tmp['b'+str(ii)+'sgd']=[]

    for x, y in data.get_data_batches(batchsize=batchsize, split='train[:1%]'):
        randkey, _ = random.split(randkey)
        _, gradsgd, optimstate = optim.sgdupdate(x, y, params, randkey, optimstate)
        _, gradnp, optimstate = optim.npupdate(x, y, params, randkey, optimstate)
        cnt=0
        for (dwtrue, dbtrue), (dwsgd, dbsgd), (dwnp, dbnp) in zip(gradtrue, gradsgd, gradnp):            
            dwtrue=np.ndarray.flatten(np.asarray(dwtrue))
            dbtrue=np.ndarray.flatten(np.asarray(dbtrue))

            dwsgd=np.ndarray.flatten(np.asarray(dwsgd))
            dbsgd=np.ndarray.flatten(np.asarray(dbsgd))

            dwnp=np.ndarray.flatten(np.asarray(dwnp))
            dbnp=np.ndarray.flatten(np.asarray(dbnp))

            tmp['w'+str(cnt)+'np'].append(angle_between(dwnp, dwtrue))
            tmp['b'+str(cnt)+'np'].append(angle_between(dbnp, dbtrue))
            tmp['w'+str(cnt)+'sgd'].append(angle_between(dwsgd, dwtrue))
            tmp['b'+str(cnt)+'sgd'].append(angle_between(dbsgd, dbtrue))
            cnt+=1
    
    for ii in range(len(layer_sizes)-1):
        tmp.update({'w'+str(ii)+'np' : np.mean(tmp['w'+str(ii)+'np'])})
        tmp.update({'b'+str(ii)+'np' : np.mean(tmp['b'+str(ii)+'np'])})
        tmp.update({'w'+str(ii)+'sgd' : np.mean(tmp['w'+str(ii)+'sgd'])})
        tmp.update({'b'+str(ii)+'sgd' : np.mean(tmp['b'+str(ii)+'sgd'])})
    
    print(tmp)
    return tmp

# randkey = random.PRNGKey(int(time.time()))
randkey = random.PRNGKey(0)

path = "explogs/grad_angle/"
log_expdata = False
num_epochs = 10
batchsize = 100
lr = 5e-3
# the points along the training trajectory which we want to compute angles with the gradient
gradangle_points=[0]

# build our network
layer_sizes = [data.num_pixels, 500, data.num_classes]

print("Network structure: {}".format(layer_sizes))

params = fc.init(layer_sizes, randkey)
optimizer=optim.sgdupdate
forward = fc.batchforward

optimstate = { 'lr' : lr, 't' : 0 }
ptr=0
expdata = {}

randkey, _ = random.split(randkey)

for epoch in range(1, num_epochs+1):
    start_time=time.time()
    
    train_acc, test_acc = train.compute_metrics(params, forward, data)
    train_acc = round(train_acc,3)

    if(train_acc > gradangle_points[ptr]):
        expdata.update({train_acc : get_grad_angles(params, randkey, optimstate)})
        print('time to perform angle compute : ', round((time.time()-start_time), 2), ' s')
        start_time=time.time()
        ptr+=1
        if(ptr==len(gradangle_points)):
            break

    print('EPOCH ', epoch)
    for x, y in data.get_data_batches(batchsize=batchsize, split=data.trainsplit):
        randkey, _ = random.split(randkey)
        params, grads, optimstate = optimizer(x, y, params, randkey, optimstate)

    
    epoch_time = time.time() - start_time
    print('epoch training time: {} test acc: {}'.format(round(epoch_time,2), round(test_acc, 3)))


# save out results of experiment
if(log_expdata):
    pickle.dump(expdata, open(path + "grad_angles_hl" + str(len(layer_sizes)-2) + ".pickle", "wb"))