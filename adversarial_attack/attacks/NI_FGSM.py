import torch.nn as nn
import torch
from tqdm import tqdm
import logging

# NI_FGSM攻击
def NI_FGSM(model, x, y, grad):
    eps = 0.1     # 扰动  
    num_iter = 10     # 迭代次数  10
    alpha = eps / num_iter        # 阿尔法
    momentum = 1.0     # 动量 1.0            

    x_nes = x + momentum * alpha * grad   

    # 输入x_nes,y_true,算损失
    x_nes.retain_grad()
    y_nes = model(x_nes)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(y_nes, y)
  
    # 损失函数的梯度
    model.zero_grad() 
    loss.backward()
    data_grad = x_nes.grad.data

    # 损失函数的梯度，除以其L1范数（各个元素的绝对值之和）
    noise = data_grad / torch.norm(data_grad, p=1)
    noise = momentum * grad + noise  # 扰动   
    noise = alpha * noise.sign()

    # x(t+1)adv
    noise = torch.clamp(noise, -0.1, 0.1)
    x = x + noise   # 对抗样本生成  

    return model, x, y, noise



def generate_adv_ex_nifgsm( model, device, test_loader, epsilon ):

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
       
        # 使用 NI_FGSM 攻击
        i = 0
        num_iter = 10     # 10
        while i <= num_iter:  # 循环了11次，加上之前的反向传播，相当于迭代了12次。可以设置为i<9,即迭代了10次
            model, x, y, data_grad = NI_FGSM(model, x, y, data_grad)
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
    logging.info("NI_FGSM: Epsilon: {}\tTest Initial Accuracy = {} / {} = {}".format(epsilon, init_correct, total, ini_acc))
    logging.info("NI_FGSM: Epsilon: {}\tTest Final Accuracy = {} / {} = {}".format(epsilon, final_correct, total, final_acc))

    # 返回准确率和对抗样本
    return final_acc, adv_examples