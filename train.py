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
from models.R2D2_embedding import R2D2Embedding
from model.protonet import ProtoNetEmbedding
from models.ResNet12_embedding import resnet12

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
    elif options.network == 'R2D2':
        network = R2D2Embedding().cuda()
    elif options.network == 'ResNet':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5).cuda()
            network = torch.nn.DataParallel(network, device_ids=[0])
        else:
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
    else:
        logging.error("Cannot recognize the network type")
        assert False

    # Choose the classification head
    if options.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()
    elif options.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif options.head == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2').cuda()
    elif options.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-He').cuda()
    else:
        logging.error("Cannot recognize the dataset type")
        assert False

    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_train = MiniImageNet(phase='train')
        dataset_val = MiniImageNet(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_train = tieredImageNet(phase='train')
        dataset_val = tieredImageNet(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_train = CIFAR_FS(phase='train')
        dataset_val = CIFAR_FS(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'FC100':
        from data.FC100 import FC100, FewShotDataloader
        dataset_train = FC100(phase='train')
        dataset_val = FC100(phase='val')
        data_loader = FewShotDataloader
    else:
        logging.error("Cannot recognize the dataset type")
        assert False

    return (dataset_train, dataset_val, data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=60, help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10, help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=15, help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=5, help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=6, help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=2000, help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15, help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5, help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=5, help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', default='./experiments/exp_1')
    parser.add_argument('--gpu', default='0, 1, 2, 3')
    parser.add_argument('--network', type=str, default='ProtoNet', help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='ProtoNet', help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet', help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=8, help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0, help='epsilon of label smoothing')

    opt = parser.parse_args()

    (dataset_train, dataset_val, data_loader) = get_dataset(opt)

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot,  # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=2,
        epoch_size=opt.episodes_per_batch * 1000,  # num of batches per epoch
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = setup_logger(opt.save_path)  # Initialize logger
    logging.info(str(vars(opt)))

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=opt.save_path)

    (embedding_net, cls_head) = get_model(opt)

    optimizer = torch.optim.SGD(
        [{'params': embedding_net.parameters()}, {'params': cls_head.parameters()}],
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )

    lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    for epoch in range(1, opt.num_epoch + 1):
        # Train on the training split
        lr_scheduler.step()

        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']

        logging.info(f"Train Epoch: {epoch}\tLearning Rate: {epoch_learning_rate:.4f}")

        _, _ = [x.train() for x in (embedding_net, cls_head)]

        train_accuracies = []
        train_losses = []

        for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

            train_n_support = opt.train_way * opt.train_shot
            train_n_query = opt.train_way * opt.train_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)

            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot)

            smoothed_one_hot = one_hot(labels_query.reshape(-1), opt.train_way)
            smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (opt.train_way - 1)

            log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way), dim=1)
            loss = -(smoothed_one_hot * log_prb).sum(dim=1)
            loss = loss.mean()

            acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))

            train_accuracies.append(acc.item())
            train_losses.append(loss.item())

            if i % 100 == 0:
                train_acc_avg = np.mean(np.array(train_accuracies))
                logging.info(
                    f"Train Epoch: {epoch}\tBatch: [{i}/{len(dloader_train)}]\tLoss: {loss.item():.4f}\tAccuracy: {train_acc_avg:.2f} % ({acc:.2f} %)"
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log training metrics to TensorBoard
        writer.add_scalar('Train/Loss', np.mean(train_losses), epoch)
        writer.add_scalar('Train/Accuracy', np.mean(train_accuracies), epoch)

        # Evaluate on the validation split
        _, _ = [x.eval() for x in (embedding_net, cls_head)]

        val_accuracies = []
        val_losses = []

        for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

            test_n_support = opt.test_way * opt.val_shot
            test_n_query = opt.test_way * opt.val_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, test_n_support, -1)
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot)

            loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())

        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        # Log validation metrics to TensorBoard
        writer.add_scalar('Validation/Loss', val_loss_avg, epoch)
        writer.add_scalar('Validation/Accuracy', val_acc_avg, epoch)

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save(
                {'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},
                os.path.join(opt.save_path, 'best_model.pth'),
            )
            logging.info(
                f"Validation Epoch: {epoch}\t\t\tLoss: {val_loss_avg:.4f}\tAccuracy: {val_acc_avg:.2f} ± {val_acc_ci95:.2f} % (Best)"
            )
        else:
            logging.info(
                f"Validation Epoch: {epoch}\t\t\tLoss: {val_loss_avg:.4f}\tAccuracy: {val_acc_avg:.2f} ± {val_acc_ci95:.2f} %"
            )

        torch.save(
            {'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},
            os.path.join(opt.save_path, 'last_epoch.pth'),
        )

        if epoch % opt.save_epoch == 0:
            torch.save(
                {'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},
                os.path.join(opt.save_path, f'epoch_{epoch}.pth'),
            )

        logging.info(f"Elapsed Time: {timer.measure()}/{timer.measure(epoch / float(opt.num_epoch))}\n")

    # Close the TensorBoard writer
    writer.close()