from jittor.dataset import Dataset
import os
import scipy.io as sio
import numpy as np
import jittor as jt
jt.flags.use_cuda = 1

class ModelSignDataset(Dataset):
    def __init__(self, mat_dir, args, is_test):
        super(ModelSignDataset, self).__init__()
        print('loading data to ram...')
        models_sign = []
        models_input = []
        self.all_names = []
        f_list = os.listdir(mat_dir)
        count = 0
        idx = -1
        for filename in f_list:
            idx = idx + 1
            print(count)
            if count > 20000:  # control the num of data used
                break

            thisname = mat_dir + os.path.splitext(filename)[0] + '.mat'
            print(os.path.splitext(filename)[0][0:3])

            if os.path.splitext(filename)[1] != '.mat':
                continue
            print(thisname)
            # print(os.path.splitext(filename)[0])
            temp_evecs = sio.loadmat(thisname)['model_evecs'][:, :args.num_evecs]
            temp_sign = sio.loadmat(thisname)['model_sign']

            file_pos = np.zeros((args.num_point, 3))
            file_evecs = np.zeros((args.num_point, args.num_evecs))
            file_sign = temp_sign

            this_point = np.shape(temp_evecs)[0]
            if this_point > args.num_point:
                file_evecs = temp_evecs[:args.num_point, :]

            else:
                file_evecs[:this_point, :] = temp_evecs

            for iEig in range(min(args.num_evecs, np.shape(file_evecs)[1])):
                # print(np.shape(np.abs(file_evecs[:,1:4])), np.shape(file_evecs[:,iEig:iEig+1]))
                # print(np.shape(np.concatenate([file_evecs[:,1:3],file_evecs[:,iEig:iEig+1]],axis=1)))
                # print(np.shape(file_evecs))
                if is_test == False:
                    if file_sign[0, iEig] == 0:
                        continue
                models_input.append(np.concatenate([file_evecs[:, 0:3], file_evecs[:, iEig:iEig + 1]], axis=1))
                models_sign.append((file_sign[0, iEig] + 1) / 2)
            self.all_names.append(thisname)
            count = count + 1
        self.num_models = count
        self.all_models_sign = np.concatenate([np.expand_dims(x, 0) for x in models_sign], axis=0)#.astype(dtype=np.float32)
        # all_models_sign = np.concatenate([np.resize(x[:,:FLAGS.num_evecs],(FLAGS.num_evecs)) for x in models_sign],axis=0)
        print(np.shape(self.all_models_sign))
        self.all_models_input = np.concatenate([np.expand_dims(x, 0) for x in models_input], axis=0)#.astype(dtype=np.float32)
        # print(all_names)

    def __len__(self):
        return len(self.all_models_sign)

    def __getitem__(self, idx):
        model_sign = self.all_models_sign[idx]
        model_input = self.all_models_input[idx]
        return model_sign, model_input


class ModelSignDemoDataset(Dataset):
    def __init__(self, mat_dir, args):
        super(ModelSignDemoDataset, self).__init__()
        print('loading data to ram...')
        models_input = []
        self.all_names = []
        f_list = os.listdir(mat_dir)
        count = 0
        idx = -1
        for filename in f_list:
            idx = idx + 1
            print(count)
            if count > 20000:  # control the num of data used
                break

            thisname = mat_dir + os.path.splitext(filename)[0] + '.mat'
            print(os.path.splitext(filename)[0][0:3])

            if os.path.splitext(filename)[1] != '.mat':
                continue
            print(thisname)
            temp_evecs = sio.loadmat(thisname)['model_evecs'][:, :args.num_evecs]

            file_evecs = np.zeros((args.num_point, args.num_evecs))

            this_point = np.shape(temp_evecs)[0]
            if this_point > args.num_point:
                file_evecs = temp_evecs[:args.num_point, :]

            else:
                file_evecs[:this_point, :] = temp_evecs

            for iEig in range(min(args.num_evecs, np.shape(file_evecs)[1])):
                models_input.append(np.concatenate([file_evecs[:, 0:3], file_evecs[:, iEig:iEig + 1]], axis=1))

            self.all_names.append(thisname)
            count = count + 1
        self.num_models = count
        self.all_models_input = np.concatenate([np.expand_dims(x, 0) for x in models_input], axis=0)#.astype(dtype=np.float32)
        # print(all_names)

    def __len__(self):
        return len(self.all_models_input)

    def __getitem__(self, idx):
        model_input = self.all_models_input[idx]
        return model_input