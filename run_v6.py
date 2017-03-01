import os
import training_l1
import training_l2
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
pfc = 1
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
lambda1_list = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
lambda2_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
lambda2 = 0.0005
dropout_rate = 1

for elem in lambda2_list:
    save_name = 'tmp' + str(lambda1_list.index(elem)) + '.pkl'
    param = [
    ('-pcov',pcov),
    ('-pcov2',pcov2),
    ('-pfc',pfc),
    ('-pfc2',pfc2),
    ('-m',model_tag),
    ('-lr',lr),
    ('-norm1',0),
    ('-norm2',elem),
    ('-dropout',dropout_rate),
    ('-train',True),
    ('-weight_file_name', save_name)
    ]
    acc = training_l1.main(param)
    model_tag = 'pcov'+str(pcov)+'pcov'+str(pcov2)+'pfc'+str(pfc)+'pfc'+str(pfc2)
    acc_list.append(acc)

print (acc_list)

print('accuracy summary: {}'.format(acc_list))
