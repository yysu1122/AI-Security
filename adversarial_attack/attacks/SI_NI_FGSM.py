import torch.nn as nn
import torch
from tqdm import tqdm
import logging

def SI_NI_FGSM(model, x, y,  grad): 
    # 原算法中m=5,因此重复到1/16
    eps = 0.1     # 扰动  
    num_iter = 10     # 迭代次数  10
    alpha = eps / num_iter        # 阿尔法
    momentum = 1.0     # 动量 1.0            

    x_nes = x + momentum * alpha * grad   # 

    # 输入x_nes,y_true,算损失函数，然后求梯度
    x_nes.retain_grad()
    y_nes = model(x_nes)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(y_nes, y)

    # 损失函数的梯度
    model.zero_grad() 
    loss.backward()
    data_grad = x_nes.grad.data

    x_nes_2 = 1 / 2 * x_nes
    # 输入x_nes_2,y_true,算损失函数，然后求梯度
    x_nes_2.retain_grad()
    y_nes_2 = model(x_nes_2)
    loss = criterion(y_nes_2, y)
    
    # 损失函数的梯度
    model.zero_grad()
    loss.backward()
    data_grad += x_nes_2.grad.data

    x_nes_4 = 1 / 4 * x_nes
    x_nes_4.retain_grad()
    y_nes_4 = model(x_nes_4)
    loss = criterion(y_nes_4, y)
    
    # 损失函数的梯度
    model.zero_grad() 
    loss.backward()
    data_grad += x_nes_4.grad.data


    x_nes_8 = 1 / 8 * x_nes
    # 输入x_nes_8,y_true,算损失函数，然后求梯度
    x_nes_8.retain_grad()
    y_nes_8 = model(x_nes_8)
    loss = criterion(y_nes_8, y)
    
    model.zero_grad() 
    loss.backward()
    data_grad += x_nes_8.grad.data


    x_nes_16 = 1 / 16 * x_nes
    # 输入x_nes_16,y_true,算损失函数，然后求梯度
    x_nes_16.retain_grad()
    y_nes_16 = model(x_nes_16)
    loss = criterion(y_nes_16, y)
    
    # 损失函数的梯度
    model.zero_grad() 
    loss.backward()
    data_grad += x_nes_16.grad.data

    data_grad = 1/5 * data_grad # 求平均******通过下一步1/5抵消了，所以可以省略这步计算

    # 损失函数的梯度，除以其L1范数
    noise = data_grad / torch.norm(data_grad, p=1)

    # g(t+1)
    noise = momentum * grad + noise  # 扰动   
    noise = alpha * noise.sign()

    # x(t+1)adv
    noise = torch.clamp(noise, -0.1, 0.1)
    x = x + noise
    
    return model, x, y, noise


def generate_adv_ex_sinifgsm( model, device, test_loader, epsilon ):

    # 记录准确率
    total = 0
    init_correct = 0
    final_correct = 0
    eps = 0.1
    adv_examples = []

    # 在测试集上循环所有的样本
    for inputs, labels in tqdm(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.squeeze(1)  #对标签列表进行降维

        # 设置张量的 requires_grad 
        inputs.requires_grad = True
        x = inputs
        y = labels
        
        # 前向传递，预测输出
        outputs = model(inputs)

        # 计算原始准确率
        _, ini_predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        init_correct += (ini_predicted == labels).sum().item()

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        # 梯度置零
        model.zero_grad()

        # 反向传播
        loss.backward()

        # 收集梯度
        data_grad = inputs.grad.data 

        # 使用 SI_NI_FGSM 攻击
        i = 0
        num_iter = 9     # 迭代次数 10 (见NI_FGSM.py解释)

        while i < num_iter:
            model, x, y, data_grad = SI_NI_FGSM(model, x, y, data_grad)
            i += 1
       
        perturbed_datas = x

        # 重新分类扰动数据
        outputs = model(perturbed_datas)

        # 检查成功率
        _, fin_predicted = torch.max(outputs.data, 1)
        final_correct += (fin_predicted == labels).sum().item()
        
        adv_ex = perturbed_datas.squeeze().detach().cpu().numpy()
        adv_examples.append(adv_ex) 

    # 为扰动计算最终的准确率
    ini_acc = init_correct/total
    final_acc = final_correct/total
    logging.info("SI_NI_FGSM: Epsilon: {}\tTest Initial Accuracy = {} / {} = {}".format(epsilon, init_correct, total, ini_acc))
    logging.info("SI_NI_FGSM: Epsilon: {}\tTest Final Accuracy = {} / {} = {}".format(epsilon, final_correct, total, final_acc))

    # 返回准确率和对抗样本
    return final_acc, adv_examples