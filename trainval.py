'''
使用POCUS数据进行分类
'''

import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models  # 有预训练好的参数
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataloader import COVIDDataset


def main(fold):
    # ============================ step 1/5 数据 ============================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.8, 1.25)), # 这里的crop size要保持与resize一致
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.18,0.18,0.18], std=[0.24,0.24,0.24])
    ])
    train_data = COVIDDataset(data_dir, fold, original_split, True, train_transform)
    valid_data = COVIDDataset(data_dir, fold, original_split, False, valid_transform)

    # 构建DataLoder，使用实例化后的数据集作为dataset
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, num_workers=0)
    
    # ============================ step 2/5 模型 ============================
    model = models.resnet18(pretrained=True)
    
    # 导入预训练参数
    if USCL_pretrain:
        state_dict = torch.load(state_dict_path)
        state_dict = {k:state_dict[k] for k in list(state_dict.keys()) if not (k.startswith('l') or k.startswith('fc'))} # 去掉2层MLP的参数
        state_dict = {k:state_dict[k] for k in list(state_dict.keys()) if not k.startswith('classifier')} # 去掉classifier的参数
        
        con_layer_names = list(state_dict.keys())
        target_layer_names = list(model.state_dict().keys())
        new_dict = {target_layer_names[i]:state_dict[con_layer_names[i]] for i in range(len(con_layer_names))}
    
        model_dict = model.state_dict()
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        print('\nThe self-supervised trained parameters are loaded.\n')
        
    for name, param in model.named_parameters():
        if not name.startswith('layer4.1'):  # 除了最后的层，其他全部固定
            param.requires_grad = False
            
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model = model.to(device)
    
    for name, param in model.named_parameters():
        print(name, '\t', 'requires_grad=', param.requires_grad)
    
    # ============================ step 3/5 损失函数 ============================
    criterion = torch.nn.CrossEntropyLoss()
    
    # ============================ step 4/5 优化器 ============================
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=MAX_EPOCH,
                                                     eta_min=0,
                                                     last_epoch=-1)
    
    # ============================ step 5/5 训练 ============================
    print('\nTraining start!\n')
    start = time.time()
    train_curve = list()
    valid_curve = list()
    max_acc = 0.
    reached = 0 # which epoch reached the max accuracy

    # the statistics of classification result: classification_results[true][pred]
    classification_results = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    best_classification_results = None
    
    for epoch in range(1, MAX_EPOCH + 1):

        loss_mean = 0.
        correct = 0.
        total = 0.

        model.train()
        for i, data in enumerate(train_loader):

            # forward
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().squeeze().sum().numpy()

            # 打印训练信息
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i+1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.

        print('Learning rate this epoch:', round(scheduler.get_lr()[0],6))
        scheduler.step()  # 更新学习率

        # validate the model
        if epoch % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            model.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).cpu().squeeze().sum().numpy()
                    for k in range(len(predicted)):
                        classification_results[labels[k]][predicted[k]] += 1 # "label" is regarded as "predicted"

                    loss_val += loss.item()

                acc = correct_val / total_val
                if acc > max_acc: # 更新最大正确率以及相应的分类结果
                    max_acc = acc
                    reached = epoch
                    best_classification_results = classification_results
                classification_results = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                valid_curve.append(loss_val/valid_loader.__len__())
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}\n".format(
                    epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val, acc))

    print('\nTraining finish, the time consumption of {} epochs is {}s\n'.format(MAX_EPOCH, round(time.time() - start)))
    print('The max validation accuracy is: {:.2%}, reached at epoch {}.\n'.format(max_acc, reached))

    print('\nThe best prediction results of the dataset:')
    print('Class 0 predicted as class 0:', best_classification_results[0][0])
    print('Class 0 predicted as class 1:', best_classification_results[0][1])
    print('Class 0 predicted as class 2:', best_classification_results[0][2])
    print('Class 1 predicted as class 0:', best_classification_results[1][0])
    print('Class 1 predicted as class 1:', best_classification_results[1][1])
    print('Class 1 predicted as class 2:', best_classification_results[1][2])
    print('Class 2 predicted as class 0:', best_classification_results[2][0])
    print('Class 2 predicted as class 1:', best_classification_results[2][1])
    print('Class 2 predicted as class 2:', best_classification_results[2][2])

    try: # 可能遇到分母为零的情况
        acc0 = best_classification_results[0][0] / sum(best_classification_results[i][0] for i in range(3))
        recall0 = best_classification_results[0][0] / sum(best_classification_results[0])
        print('\nClass 0 accuracy:', acc0)
        print('Class 0 recall:', recall0)
        print('Class 0 F1:', 2 * acc0 * recall0 / (acc0 + recall0))

        acc1 = best_classification_results[1][1] / sum(best_classification_results[i][1] for i in range(3))
        recall1 = best_classification_results[1][1] / sum(best_classification_results[1])
        print('\nClass 1 accuracy:', acc1)
        print('Class 1 recall:', recall1)
        print('Class 1 F1:', 2 * acc1 * recall1 / (acc1 + recall1))

        acc2 = best_classification_results[2][2] / sum(best_classification_results[i][2] for i in range(3))
        recall2 = best_classification_results[2][2] / sum(best_classification_results[2])
        print('\nClass 2 accuracy:', acc2)
        print('Class 2 recall:', recall2)
        print('Class 2 F1:', 2 * acc2 * recall2 / (acc2 + recall2))
    except:
        print('meet 0 denominator')
    
    return best_classification_results


if __name__ == '__main__':
    # 设置超参数
    meta = True
    data_dir = 'D:/数据集/POCUS_5fold'
    USCL_pretrain = False
    state_dict_path = 'best_model.pth'  # 预训练参数
    lr = 1e-3
    weight_decay = 1e-4
    original_split = False  # 原始的数据分割只有50%上下准确率，这是由于不同fold来源不同，不同分布
    MAX_EPOCH = 30
    BATCH_SIZE = 128
    log_interval = 4
    val_interval = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    confusion_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for i in range(5):
        print('\n' + '='*20 + 'The training of fold {} start.'.format(i) + '='*20)
        best_classification_results = main(i)
        confusion_matrix = confusion_matrix + np.array(best_classification_results)
    
    # 计算一些metrics并显示、保存
    prec_COVID = confusion_matrix[0,0] / sum(confusion_matrix[:,0])
    prec_Pneu = confusion_matrix[1,1] / sum(confusion_matrix[:,1])
    prec_Reg = confusion_matrix[2,2] / sum(confusion_matrix[:,2])
    sen_COVID = confusion_matrix[0,0] / sum(confusion_matrix[0])
    sen_Pneu = confusion_matrix[1,1] / sum(confusion_matrix[1])
    sen_Reg = confusion_matrix[2,2] / sum(confusion_matrix[2])
    spe_COVID = confusion_matrix[1:,1:].sum() / confusion_matrix[1:].sum()
    spe_Pneu = confusion_matrix[[0,2]][:,[0,2]].sum() / confusion_matrix[[0,2]].sum()
    acc = (confusion_matrix[0,0]+confusion_matrix[1,1]+confusion_matrix[2,2])/confusion_matrix.sum()
    acc = round(acc, 4) * 100
    
    print('\nThe confusion matrix is:')
    print(confusion_matrix)
    print('\nThe precision of COVID is:', prec_COVID)
    print('The precision of Pneumonia is:', prec_Pneu)
    print('The precision of Regular is:', prec_Reg)
    print('\nThe recall (sensitivity) of COVID is:', sen_COVID)
    print('The recall (sensitivity) of Pneumonia is:', sen_Pneu)
    print('The recall (sensitivity) of Regular is:', sen_Reg)
    print('\nThe specificity of COVID is:', spe_COVID)  # 没患新冠的人被诊断为没有患的比例
    print('The specificity of Pneumonia is:', spe_Pneu)
    print('\nTotal acc is:', acc)

    # write logs
    with open('finetune_POCUS.txt', 'a') as f:
        f.write('\nThe confusion matrix is:\n' + str(confusion_matrix))
        f.write('\nThe precision of COVID is: ' + str(prec_COVID))
        f.write('\nThe precision of Pneumonia is: ' + str(prec_Pneu))
        f.write('\nThe precision of Regular is: ' + str(prec_Reg))
        f.write('\nThe recall (sensitivity) of COVID is: ' + str(sen_COVID))
        f.write('\nThe recall (sensitivity) of Pneumonia is: ' + str(sen_Pneu))
        f.write('\nThe recall (sensitivity) of Regular is: ' + str(sen_Reg))
        f.write('\nThe specificity of COVID is: ' + str(spe_COVID))
        f.write('\nThe specificity of Pneumonia is: ' + str(spe_Pneu))
        f.write('\nTotal acc is: ' + str(acc) + '\n\n\n')




