import torch.nn as nn
import torch
from tqdm import tqdm
import logging
## 参考：https://pytorch.org/tutorials/beginner/fgsm_tutorial.html?highlight=fgsm

# FGSM-对网络流量的扰动攻击
def fgsm_attack(traffic_data, epsilon, data_grad):
    
    sign_data_grad = data_grad.sign()  # 收集数据梯度的逐元素符号
    perturbed_traffic_data = traffic_data + epsilon*sign_data_grad # 利用原始流量数据生成扰动流量数据
    
    return perturbed_traffic_data  # 返回扰动流量数据

def generate_adv_ex_fgsm( model, device, criterion,test_loader, epsilon ):

    # 记录准确率
    total = 0
    init_correct = 0
    final_correct = 0
    adv_examples = []

    # 循环测试集所有样本
    for inputs, labels in tqdm(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.squeeze(1)  #对标签列表进行降维。squeeze(1)代表若第二维度值为1则去除第二维度

        # Set requires_grad attribute of tensor. Important for Attack
        inputs.requires_grad = True
        
        # Forward pass the data through the model
        outputs = model(inputs)

        # 计算原始准确率
        # _, ini_predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        # init_correct += (ini_predicted == labels).sum().item()

        # If the initial prediction is wrong, dont bother attacking, just move on
        # print(init_pred)
        # print(labels)
        # if init_pred.item() != labels:
        #      continue

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = inputs.grad.data

        # Call FGSM Attack
        perturbed_datas = fgsm_attack(inputs, epsilon, data_grad)

        # Re-classify the perturbed image
        outputs = model(perturbed_datas)

        # Check for success
        _, fin_predicted = torch.max(outputs.data, 1)
        final_correct += (fin_predicted == labels).sum().item()
        
        adv_ex = perturbed_datas.squeeze().detach().cpu().numpy()
        adv_examples.append(adv_ex) 

    # Calculate final accuracy for this epsilon
    final_acc = final_correct/total
    logging.info("FGSM: Epsilon: {}\tTest Final Accuracy = {} / {} = {}".format(epsilon, final_correct, total, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples