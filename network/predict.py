import os
import time
import jittor as jt
import pdb
import argparse
import numpy as np
import scipy.io as sio

# TODO: Why lmx do not use this
jt.flags.use_cuda = 1

from models import SignNet
from dataset import ModelSignDataset, ModelSignDemoDataset
from tensorboardX import SummaryWriter
from tqdm import tqdm

# test_ratio = 7
# high_eval_acc = 0


def predict_one_epoch(pre_loader, model, epoch, writer, FLAGS):
    model.eval()

    ss_ours = []

    pre_loader_iter = iter(pre_loader)
    for model_i in range(pre_loader.num_models):
        ss_our = []
        num_batch = FLAGS.num_evecs // FLAGS.eval_batch_size
        for j in range(num_batch):
            model_input = next(pre_loader_iter)  # TODO: check next? what if reset?
            # TODO: set eval num_worker to 1?
            # TODO: do I need jt.no_grad?
            pre_sign = model(model_input)

            ss_our.append(pre_sign)
        ss_ours.append(ss_our)

    return ss_ours # return gt our to store in mat


def main():
    parser = argparse.ArgumentParser()
    # training params
    parser.add_argument("--learning_rate", type=float, default=1e-4, help='initial learning rate.')
    parser.add_argument("--batch_size", type=int, default=1, help='batch size')
    parser.add_argument("--eval_batch_size", type=int, default=1, help='batch size')
    parser.add_argument("--num_point", type=int, default=4500, help='num of point')

    # architecture parameters
    parser.add_argument("--num_evecs", type=int, default=17, help='number of eigenvectors used for representation. '
                                                                  'The first 500 are precomputed and stored in input')
    parser.add_argument("--dim_input", type=int, default=4, help='')
    # parser.add_argument("--dim_constraint", type=int, default=40)

    # data parameters
    parser.add_argument("--pred_dir", type=str, default='../data/demo/')
    parser.add_argument("--save_root", type=str, default='./Results/train_inter_k_flag',
                        help='directory to save models and results')
    parser.add_argument('--save_freq', type=int, default=30,
                        help='save frequency')
    parser.add_argument("--max_train_iter", type=int, default=150)
    # parser.add_argument("--save_summaries_secs", type=int, default=60)
    # parser.add_argument("--save_model_secs", type=int, default=1200)
    # parser.add_argument("--master", type=str, default='')

    FLAGS = parser.parse_args()
    assert FLAGS.num_evecs % FLAGS.eval_batch_size == 0, "eval_batch_size can not divide num_evecs exactly"
    print('save_root=%s' % FLAGS.save_root)  # root of save
    if not os.path.isdir(FLAGS.save_root):
        os.makedirs(FLAGS.save_root)
    FLAGS.model_root_path = os.path.join(FLAGS.save_root, 'models')
    FLAGS.log_root_path = os.path.join(FLAGS.save_root, 'logs')
    FLAGS.model_name = "IntrinsicSym"
    FLAGS.model_folder = os.path.join(FLAGS.model_root_path, FLAGS.model_name)
    FLAGS.log_folder = os.path.join(FLAGS.log_root_path, FLAGS.model_name)
    FLAGS.log_file_path = os.path.join(FLAGS.log_folder, FLAGS.model_name + '.log')

    if not os.path.isdir(FLAGS.model_folder):
        os.makedirs(FLAGS.model_folder)
    if not os.path.isdir(FLAGS.log_folder):
        os.makedirs(FLAGS.log_folder)
    print('num_evecs=%d' % FLAGS.num_evecs)

    model = SignNet(FLAGS.num_point, FLAGS.dim_input)

    pred_dataset = ModelSignDemoDataset(FLAGS.pred_dir, args=FLAGS).set_attrs(batch_size=FLAGS.batch_size,
                                                                                        shuffle=False, num_workers=4,
                                                                                        drop_last=True)

    writer = SummaryWriter(log_dir=FLAGS.log_folder)

    print('starting session...')
    iteration = 0

    eval_best_ss = np.zeros((pred_dataset.num_models, FLAGS.num_evecs))

    start_time = time.time()
    for _ in range(1):
        ss_ours = predict_one_epoch(pred_dataset, model, iteration, writer, FLAGS)
    duration = time.time() - start_time
    print("duration is ")
    print(duration)
    for model_i in range(len(ss_ours)):
        if True:  # acc_per_model[model_i] > best_acc[model_i]:
            batch_num = len(ss_ours[model_i])
            batch_size = FLAGS.eval_batch_size
            ss_our = ss_ours[model_i]
            for j in range(batch_num):
                eval_best_ss[model_i, j * batch_size:(j + 1) * batch_size] = ss_our[j].numpy()
    sio.savemat('../data/predict/S.mat', {'name': pred_dataset.all_names, 'ss_our': eval_best_ss})


if __name__ == "__main__":
    main()
