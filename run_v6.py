import os
import training_l1
import compute_lambda
import sys
# os.system('python training_v3.py -p0')
# os.system('python training_v3.py -p1')
# os.system('python training_v3.py -p2')
# os.system('python training_v3.py -p3')
# os.system('python training_v3.py -p4')
# os.system('python training_v3.py -p4')
# os.system('python training_v3.py -p5')

acc_list = []
count = 0
pcov = 0
pfc = 0
pcov2 = 0
pfc2 = 0
# model_tag = 'pcov'+str(pcov)+'pcov'+str(pcov2)+'pfc'+str(pfc)+'pfc'+str(pfc2)
# pfc = pfc+1
# param = [
# ('-pcov',pcov),
# ('-pcov2',pcov2),
# ('-pfc',pfc),
# ('-pfc2',pfc2),
# ('-m',model_tag),
# ('-ponly', True),
# ('-test', False)
# ]
# acc = training_v6.main(param)
retrain = 0
lr = 1e-4
model_tag = 'pcov'+str(pcov)+'pcov'+str(pcov2)+'pfc'+str(pfc)+'pfc'+str(pfc2)
lambda1_list = [1e-3]
# lambda1_list = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
# lambda2_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
# lambda2_list = [1e-4]
dropout_rate_list= [0.2, 0.4, 0.6, 0.8, 1]
shakeout_rate_list = [0.2, 0.4, 0.6, 0.8, 1]
dropout_rate_list= [0.8]
c_list = [0.5,1,2,4,10]
lambda2 = 0.0005
dropout_rate = 1
parent_dir = './weights/norm1/'

# for shake_rate in shakeout_rate_list:
for lnorm1 in lambda1_list:
    # save_name = 'norm1'+'val'+str(0) +'.pkl'
    save_name = 'start'+'.pkl'
    # save_name = 'tmp' + str(shakeout_rate_list.index(shake_rate) + 1) + '.pkl'
    # save_name = 'tmp' + '.pkl'
    fetch_lambdas_params = [
        ('-PREV_EXIST',0),
        ('-parent_dir',parent_dir),
        ('-file_name', save_name)
    ]
    l1, l2 = compute_lambda.main(fetch_lambdas_params)
    l1 = 1e-8
    l2 = 1e-4
    # l1 = 0
    # l2 = 0
    print('picked l1 l2 to be {},{}'.format(l1,l2))
    # sys.exit()
    param = [
    ('-pcov',pcov),
    ('-pcov2',pcov2),
    ('-pfc',pfc),
    ('-pfc2',pfc2),
    ('-m',model_tag),
    ('-lr',lr),
    ('-norm1',l1),
    ('-norm2',l2),
    ('-dropout', 1.),
    ('-train',True),
    ('-weight_file_name', save_name),
    ('-shakeout_c', 10.),
    ('-parent_dir', parent_dir)
    ]
    acc = training_l1.main(param)
    model_tag = 'pcov'+str(pcov)+'pcov'+str(pcov2)+'pfc'+str(pfc)+'pfc'+str(pfc2)
    acc_list.append(acc)

print (acc_list)

print('accuracy summary: {}'.format(acc_list))
