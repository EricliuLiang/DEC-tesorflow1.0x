import os
import argparse
import numpy as np
from datasets import load_data
# from datasets import DataLoader
import tensorflow as tf
from time import time
from Train import Pretrainer
from Train import Trainer

def args():
    # 参数
    parser = argparse.ArgumentParser(description='Super Parameters of Model')
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int, help='Maximum Iterations')
    parser.add_argument('--tol', default=0.001, type=float, help='Iterations Stop criterion')
    parser.add_argument('--pretrained_ae_ckpt_path', default='./ae_ckpt/model.ckpt')
    parser.add_argument('--pretrain_epochs', default=None, type=int, help='Number of epochs for pretrain')
    parser.add_argument('--update_interval', default=None, type=int, help='update auxiliary target distribution p')
    parser.add_argument('--encoder_dims', default=[500, 500, 2000, 10], help='dims of encoder')
    parser.add_argument('--pretrain', default=True)
    parser.add_argument("--img_size", type=int, default=28)
    args = parser.parse_args()
    return args
def main(args):
    # 加载数据
    x, y = load_data(args.dataset)
    n_clusters = len(np.unique(y))

    # 设置参数
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        args.update_interval = 140
        args.pretrain_epochs = 301
        ae_weights_init = tf.variance_scaling_initializer(scale=1. / 3., mode='fan_in', distribution='uniform')
    # add feature dimension size to the beginning of hidden_dims
    feature_dim = x.shape[1]
    args.encoder_dims = [feature_dim] + args.encoder_dims
    print(args.encoder_dims)
    if args.pretrain == True:
        # 预训练
        print('Begin Pretraining')
        t0 = time()
        pretrainer = Pretrainer(args, ae_weights_init)
        saver = pretrainer(x, y)
        # print(saver)
        print('Pretraining time: %ds' % round(time() - t0))
    # 清理计算图
    tf.reset_default_graph()
    # Model训练
    print('Begin Model training')
    t1 = time()
    trainer = Trainer(args, ae_weights_init, n_clusters)
    trainer(x, y)
    print('Model training time: %ds' % round(time() - t1))

if __name__ == "__main__":
    # setting the hyper parameters
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    args = args()
    main(args)

