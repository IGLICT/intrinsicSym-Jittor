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
from dataset import ModelSignDataset
from tensorboardX import SummaryWriter
from tqdm import tqdm

# test_ratio = 7
# high_eval_acc = 0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(train_loader, model, optimizer, epoch, writer):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.train()
    for idx, (model_sign, model_input) in tqdm(enumerate(train_loader)):
        pre_sign, loss = model(model_input, model_sign)
        optimizer.step(loss)

        loss_meter.update(loss.data)  # TODO: check here

        my_acc = np.mean((pre_sign == model_sign).data)
        acc_meter.update(my_acc)

    writer.add_scalar("train/loss", loss_meter.avg, global_step=epoch)
    writer.add_scalar("train/acc", acc_meter.avg, global_step=epoch)


def eval_one_epoch(val_loader, model, epoch, writer, FLAGS):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.eval()

    acc_per_model = []
    ss_ours = []

    gt_our = []
    val_loader_iter = iter(val_loader)
    for model_i in tqdm(range(val_loader.num_models)):
        model_i_acc = AverageMeter()
        ss_our = []
        num_batch = FLAGS.num_evecs // FLAGS.eval_batch_size
        for j in range(num_batch):
            (model_sign, model_input) = next(val_loader_iter)  # TODO: check next? what if reset?
            # TODO: set eval num_worker to 1?
            # TODO: do I need jt.no_grad?
            pre_sign, loss = model(model_input, model_sign)

            ss_our.append(pre_sign)
            gt_our.append(model_sign)
            loss_meter.update(loss.data)  # TODO: check here, use .data?
            my_acc = np.mean((pre_sign == model_sign).data)  # TODO: .data vs .numpy?
            acc_meter.update(my_acc)
            model_i_acc.update(my_acc)
        acc_per_model.append(model_i_acc.avg)  # list, model_num * batch_num * ss_per_batch
        ss_ours.append(ss_our)
    writer.add_scalar("test/loss", loss_meter.avg, global_step=epoch)
    writer.add_scalar("test/acc", acc_meter.avg, global_step=epoch)

    return acc_per_model, ss_ours, gt_our  # return gt our to store in mat


def main():
    parser = argparse.ArgumentParser()
    # training params
    parser.add_argument("--learning_rate", type=float, default=1e-4, help='initial learning rate.')
    parser.add_argument("--batch_size", type=int, default=4, help='batch size')
    parser.add_argument("--eval_batch_size", type=int, default=4, help='batch size')
    parser.add_argument("--num_point", type=int, default=4500, help='num of point')

    # architecture parameters
    parser.add_argument("--num_evecs", type=int, default=12, help='number of eigenvectors used for representation. '
                                                                  'The first 500 are precomputed and stored in input')
    parser.add_argument("--dim_input", type=int, default=4, help='')
    # parser.add_argument("--dim_constraint", type=int, default=40)

    # data parameters
    parser.add_argument("--train_dir", type=str, default='../data/annotated/train/')
    parser.add_argument("--test_dir", type=str, default='../data/annotated/test/')
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

    train_dataset = ModelSignDataset(FLAGS.train_dir, args=FLAGS, is_test=False).set_attrs(batch_size=FLAGS.batch_size,
                                                                                           shuffle=True, num_workers=4,
                                                                                           drop_last=True)
    eval_dataset = ModelSignDataset(FLAGS.test_dir, args=FLAGS, is_test=True).set_attrs(batch_size=FLAGS.batch_size,
                                                                                        shuffle=False, num_workers=4,
                                                                                        drop_last=True)

    optimizer = jt.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    writer = SummaryWriter(log_dir=FLAGS.log_folder)

    print('starting session...')
    iteration = 0
    eval_best_acc = np.zeros(eval_dataset.num_models)
    eval_best_ss = np.zeros((eval_dataset.num_models, FLAGS.num_evecs))
    while iteration < FLAGS.max_train_iter:
        if iteration % 3 == 0:
            acc_per_model, ss_ours, gt_our = eval_one_epoch(eval_dataset, model, iteration, writer, FLAGS)
            for model_i in range(len(acc_per_model)):
                if True:  # acc_per_model[model_i] > best_acc[model_i]:
                    eval_best_acc[model_i] = acc_per_model[model_i]
                    batch_num = len(ss_ours[model_i])
                    batch_size = FLAGS.eval_batch_size
                    ss_our = ss_ours[model_i]
                    for j in range(batch_num):
                        eval_best_ss[model_i, j * batch_size:(j + 1) * batch_size] = ss_our[j].numpy()
            sio.savemat('S.mat', {'name': eval_dataset.all_names, 'ss_our': eval_best_ss, 'gt_our': gt_our,
                                  'per_acc': eval_best_acc})
        start_time = time.time()
        train_one_epoch(train_dataset, model, optimizer, iteration, writer)
        duration = time.time() - start_time
        print(duration)

        if iteration % FLAGS.save_freq == 0:
            save_file = os.path.join(
                FLAGS.model_folder, 'ckpt_epoch_{epoch}.pkl'.format(epoch=iteration))
            model.save(save_file)
        iteration += 1


if __name__ == "__main__":
    main()
