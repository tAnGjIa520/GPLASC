import torch
import numpy as np



def get_incremental_acc_util(appr,test_loader_list,max_task,device='cuda'):

    """
    模型传进来，并且返回最终的测试结果，不接受其他的测试方式
    :param model:
    :param test_loader_list:
    :param device:
    :return:
    """

    # Test

    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))

    for current_task_id in range(max_task + 1):
        test_loss, acc_taw[max_task, current_task_id], acc_tag[max_task, current_task_id] = appr.eval(current_task_id, test_loader_list[current_task_id])
        if current_task_id < max_task:
            forg_taw[max_task, current_task_id] = acc_taw[:max_task, current_task_id].max(0) - acc_taw[max_task, current_task_id]
            forg_tag[max_task, current_task_id] = acc_tag[:max_task, current_task_id].max(0) - acc_tag[max_task, current_task_id]
        print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
              '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(current_task_id, test_loss,
                                                             100 * acc_taw[max_task, current_task_id], 100 * forg_taw[max_task, current_task_id],
                                                             100 * acc_tag[max_task, current_task_id], 100 * forg_tag[max_task, current_task_id]))






def validation_util(model,validation_loader,validation_method=None,device='cuda'):
    if validation_method is None:
        validation_method=acc_util

    return validation_method(model,validation_loader)


def acc_util(model,test_loader,device='cuda'):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return 100 * correct / total
