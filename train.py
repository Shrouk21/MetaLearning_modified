# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.autograd import Variable
import logging  # Added for logging
from torch.utils.tensorboard import SummaryWriter  # Added for TensorBoard

from model.classification_head import ClassificationHead
from model.protonet import ProtoNetEmbedding


from utils import set_gpu, Timer, count_accuracy, check_dir

# Configure logging
def setup_logger(save_path):
    log_file_path = os.path.join(save_path, "train_log.txt")
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Logger initialized.")
    return log_file_path

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
    """
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    return encoded_indicies


def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
    elif options.network == 'ResNet':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5).cuda()
            network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
        else:
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
    else:
        print ("Cannot recognize the network type")
        assert(False)
        
    # Choose the classification head
    if options.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()
    elif options.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-He').cuda()
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_train = MiniImageNet(phase='train')
        dataset_val = MiniImageNet(phase='val')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_train, dataset_val, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=60)
    parser.add_argument('--save-epoch', type=int, default=10)
    parser.add_argument('--train-shot', type=int, default=15)
    parser.add_argument('--val-shot', type=int, default=5)
    parser.add_argument('--train-query', type=int, default=6)
    parser.add_argument('--val-episode', type=int, default=2000)
    parser.add_argument('--val-query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./experiments/exp_1')
    parser.add_argument('--gpu', default='0, 1, 2, 3')
    parser.add_argument('--network', type=str, default='ProtoNet')
    parser.add_argument('--head', type=str, default='ProtoNet')
    parser.add_argument('--dataset', type=str, default='miniImageNet')
    parser.add_argument('--episodes-per-batch', type=int, default=8)
    parser.add_argument('--eps', type=float, default=0.0)
    opt = parser.parse_args()

    (dataset_train, dataset_val, data_loader) = get_dataset(opt)
    dloader_train = data_loader(dataset_train, opt.train_way, 0, opt.train_shot, opt.train_way * opt.train_query, 0,
                                opt.episodes_per_batch, 4, opt.episodes_per_batch * 1000)
    dloader_val = data_loader(dataset_val, opt.test_way, 0, opt.val_shot, opt.val_query * opt.test_way, 0,
                              1, 0, opt.val_episode)

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    writer = SummaryWriter(log_dir=os.path.join(opt.save_path, "runs"))
    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)
    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                 {'params': cls_head.parameters()}],
                                lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else 0.0024)
    )

    max_val_acc = 0.0
    x_entropy = torch.nn.CrossEntropyLoss()

    for epoch in range(1, opt.num_epoch + 1):
        lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        log(log_file_path, f'\nEpoch {epoch}, Learning Rate: {lr:.5f}')
        writer.add_scalar("LR", lr, epoch)

        embedding_net.train()
        cls_head.train()
        train_loss_list = []
        train_acc_list = []

        for i, batch in enumerate(tqdm(dloader_train(epoch), desc=f"Epoch {epoch}"), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]
            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(opt.episodes_per_batch, -1, emb_support.size(-1))
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(opt.episodes_per_batch, -1, emb_query.size(-1))
            logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot)

            smoothed = one_hot(labels_query.reshape(-1), opt.train_way)
            smoothed = smoothed * (1 - opt.eps) + (1 - smoothed) * opt.eps / (opt.train_way - 1)
            log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way), dim=1)
            loss = -(smoothed * log_prb).sum(dim=1).mean()
            acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))

            train_loss_list.append(loss.item())
            train_acc_list.append(acc.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = np.mean(train_loss_list)
        avg_train_acc = np.mean(train_acc_list)
        writer.add_scalar("Train/Loss", avg_train_loss, epoch)
        writer.add_scalar("Train/Accuracy", avg_train_acc, epoch)

        log(log_file_path, f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%")

        # Optional validation step could go here
        # Add writer.add_scalar("Val/Loss", ...) and log(...) if needed

    writer.close()
