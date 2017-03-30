import os
import training_l1
import compute_lambda
import sys
acc_list = []
count = 0
pcov = 0
pfc = 0
pcov2 = 0
pfc2 = 0

parent_dir = './weights/pruning/'
def compute_file_name(pcov, pfc):
    name = ''
    name += 'cov' + str(int(pcov[0] * 10))
    name += 'cov' + str(int(pcov[1] * 10))
    name += 'fc' + str(int(pfc[0] * 10))
    name += 'fc' + str(int(pfc[1] * 10))
    return name

acc_list = []
count = 0
pcov = [0., 0.]
pfc = [90., 0.]

retrain = 0
lr = 1e-4
f_name = compute_file_name(pcov,pfc)
pfc[0] = pfc[0] + 1.

while (count < 10):
    if (retrain == 0):
        lr = 1e-4

    fetch_lambdas_params = [
        ('-PREV_EXIST',1),
        ('-parent_dir',parent_dir),
        ('-file_name', f_name + '.pkl')
    ]
    l1, l2 = compute_lambda.main(fetch_lambdas_params)

    # prune
    param = [
    ('-pcov',pcov[0]),
    ('-pcov2',pcov[1]),
    ('-pfc',pfc[0]),
    ('-pfc2',pfc[1]),
    ('-m',f_name),
    ('-weight_file_name', f_name + '.pkl'),
    ('-shakeout_c', 10.),
    ('-lr',lr),
    ('-norm1',l1),
    ('-norm2',l2),
    ('-dropout', 1.),
    ('-PRUNE',True),
    ('-TRAIN',False),
    ('-parent_dir',parent_dir)
    ]

    _ = training_l1.main(param)
    f_name = compute_file_name(pcov,pfc)

    # train pruned model
    param = [
    ('-pcov',pcov[0]),
    ('-pcov2',pcov[1]),
    ('-pfc',pfc[0]),
    ('-pfc2',pfc[1]),
    ('-m',f_name),
    ('-weight_file_name', f_name + '.pkl'),
    ('-shakeout_c', 10.),
    ('-lr',lr),
    ('-norm1',l1),
    ('-norm2',l2),
    ('-dropout', 1.),
    ('-PRUNE',False),
    ('-TRAIN',True),
    ('-parent_dir',parent_dir)
    ]
    acc,iter_cnt = training_l1.main(param)

    if (acc < 0.98710):
        retrain += 1
        lr = lr / float(2)
        if (retrain > 3):
            print("lowest precision")
            acc_list.append('{},{},{}\n'.format(
                pcov[:] + pfc[:],
                acc,
                iter_cnt
            ))
            with open("hist.txt","w") as f:
                for item in acc_list:
                    f.write(item)
            pfc[0] = pfc[0] + 1.
            if (pfc[0] == 100):
                break
            # pcov[0] = pcov[0] + 10.
    else:
        acc_list.append('{},{},{}\n'.format(
            pcov[:] + pfc[:],
            acc,
            iter_cnt
        ))
        with open("hist.txt","w") as f:
            for item in acc_list:
                f.write(item)
        pfc[0] = pfc[0] + 1.
        if (pfc[0] == 100):
            break
        # pfc[1] = pfc[1] + 10.
        # pcov[0] = pcov[0] + 10.
        count = count + 1
        if (retrain != 0):
            retrain = 0

with open("hist.txt","w") as f:
    for item in acc_list:
        f.write(item)
print('accuracy summary: {}'.format(acc_list))
