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

from model.classification_head import ClassificationHead
from model.protonet import ProtoNetEmbedding
from model.resnet import resnet12


from utils import set_gpu, Timer, count_accuracy, check_dir, log
from torch.utils.tensorboard import SummaryWriter

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()

    elif options.network == 'ResNet':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5).cuda()
            network = torch.nn.DataParallel(network, device_ids=[0, 1])
        else:
            network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
    else:
        print ("Cannot recognize the network type")
        assert(False)
        
    # Choose the classification head
    if options.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()

    elif options.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
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
    parser.add_argument('--num-epoch', type=int, default=15,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                            help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=15,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=5,
                            help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=6,
                            help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=2000, #instead of 2000
                            help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                            help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,
                            help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=5,
                            help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', default='./experiments/exp_1')
    parser.add_argument('--gpu', default='0, 1, 2, 3')
    parser.add_argument('--network', type=str, default='ProtoNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='SVM',
                            help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                            help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')
    parser.add_argument('--load', default=None,
                            help='path of the checkpoint file')
    parser.add_argument('--lr', type=float, default=0.1,
                            help='initial learning rate')
    opt = parser.parse_args()
    
    (dataset_train, dataset_val, data_loader) = get_dataset(opt)

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot, # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=4,
        epoch_size=opt.episodes_per_batch * 1000, # num of batches per epoch change back to 1000
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot, # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode, # num of batches per epoch
    )

    set_gpu(opt.gpu)

check_dir(opt.save_path)
log_file_path = os.path.join(opt.save_path, "train_log.txt")
log(log_file_path, str(vars(opt)))
print(f"TensorBoard log_dir will be: {os.path.join(opt.save_path, 'run/')}")
writer = SummaryWriter(log_dir=os.path.join(opt.save_path, 'run/'), comment='-train')
lr = opt.lr
(embedding_net, cls_head) = get_model(opt)
if opt.load:
    checkpoint = torch.load(opt.load)
    embedding_net.load_state_dict(checkpoint['embedding'])
    cls_head.load_state_dict(checkpoint['head'])
    print(f"Loaded model from {opt.load}")
optimizer = torch.optim.SGD([{'params': embedding_net.parameters()}, {'params': cls_head.parameters()}],
                            lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else 0.0024)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)
#using cosine annealing
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epoch, eta_min=1e-5)

max_val_acc = 0.0
timer = Timer()
x_entropy = torch.nn.CrossEntropyLoss()

for epoch in range(1, opt.num_epoch + 1):
    lr_scheduler.step()
    epoch_lr = optimizer.param_groups[0]['lr']
    log(log_file_path, f'Train Epoch: {epoch}\tLearning Rate: {epoch_lr:.4f}')
    embedding_net.train(); cls_head.train()

    train_accuracies, train_losses = [], []
    for i, batch in enumerate(tqdm(dloader_train(epoch), desc=f"Epoch {epoch} Training"), 1):
        data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]
        n_support, n_query = opt.train_way * opt.train_shot, opt.train_way * opt.train_query

        emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
        emb_support = emb_support.reshape(opt.episodes_per_batch, n_support, -1)
        emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
        emb_query = emb_query.reshape(opt.episodes_per_batch, n_query, -1)

        logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot)

        smoothed = one_hot(labels_query.reshape(-1), opt.train_way)
        smoothed = smoothed * (1 - opt.eps) + (1 - smoothed) * opt.eps / (opt.train_way - 1)
        log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way), dim=1)
        loss = -(smoothed * log_prb).sum(dim=1).mean()
        acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))

        train_accuracies.append(acc.item())
        train_losses.append(loss.item())

        if i % 100 == 0:
            avg_acc = np.mean(train_accuracies)
            log(log_file_path, f'Epoch {epoch} Batch {i}/{len(dloader_train)} Loss: {loss.item():.4f} Acc: {avg_acc:.2f}% ({acc:.2f}%)')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    writer.add_scalars('Train ResNet and SVM', {'Loss': np.mean(train_losses), 'Accuracy': np.mean(train_accuracies)}, epoch)
    # writer.add_scalar('Train/Loss', np.mean(train_losses), epoch)
    # writer.add_scalar('Train/Accuracy', np.mean(train_accuracies), epoch)
    writer.flush()
    

    embedding_net.eval(); cls_head.eval()
    val_accuracies, val_losses = [], []
    for i, batch in enumerate(tqdm(dloader_val(epoch), desc=f"Epoch {epoch} Validation"), 1):

        data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]
        n_support, n_query = opt.test_way * opt.val_shot, opt.test_way * opt.val_query

        emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
        emb_support = emb_support.reshape(1, n_support, -1)
        emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
        emb_query = emb_query.reshape(1, n_query, -1)

        logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot)
        loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
        acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

        val_accuracies.append(acc.item())
        val_losses.append(loss.item())

    val_acc_avg = np.mean(val_accuracies)
    val_loss_avg = np.mean(val_losses)
    val_acc_ci95 = 1.96 * np.std(val_accuracies) / np.sqrt(opt.val_episode)
    writer.add_scalars('Validation ResNet and SVM', {'Loss': val_loss_avg, 'Accuracy': val_acc_avg}, epoch)

    # Select one random image from data_support and data_query
    random_index_support = random.randint(0, data_support.shape[1] - 1)  # Random index for support images
    random_index_query = random.randint(0, data_query.shape[1] - 1)      # Random index for query images

    # Extract the selected images
    selected_support_image = data_support[0, random_index_support]  # Shape: (C, H, W)
    selected_query_image = data_query[0, random_index_query]        # Shape: (C, H, W)

    # Add the selected images to TensorBoard
    writer.add_image('Validation ResNet/Support', selected_support_image, epoch)
    writer.add_image('Validation ResNet/Query', selected_query_image, epoch)

    # writer.add_scalar('Validation/Loss', val_loss_avg, epoch)
    # writer.add_scalar('Validation/Accuracy', val_acc_avg, epoch)
    writer.flush()

    if val_acc_avg > max_val_acc:
        max_val_acc = val_acc_avg
        torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},
                   os.path.join(opt.save_path, 'best_model.pth'))
        log(log_file_path, f'Validation Epoch: {epoch}\tLoss: {val_loss_avg:.4f}\tAccuracy: {val_acc_avg:.2f} ± {val_acc_ci95:.2f}% (Best)')
    else:
        log(log_file_path, f'Validation Epoch: {epoch}\tLoss: {val_loss_avg:.4f}\tAccuracy: {val_acc_avg:.2f} ± {val_acc_ci95:.2f}%')

    torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},
               os.path.join(opt.save_path, 'last_epoch.pth'))

    if epoch % opt.save_epoch == 0:
        torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},
                   os.path.join(opt.save_path, f'epoch_{epoch}.pth'))

    log(log_file_path, f'Elapsed Time: {timer.measure()}/{timer.measure(epoch / float(opt.num_epoch))}\n')
writer.flush()
writer.close()
