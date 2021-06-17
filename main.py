import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, utils
from IPython import display
import time
from LeNet import LeNet

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 根据下标访问训练集任意一个样本
def show_one_sample(i, train_dataset, test_dataset):
    # 查看数据集的类型
    print(type(train_dataset))
    print(len(train_dataset), len(test_dataset))
    feature, label = train_dataset[i]
    print(feature.shape, label)  # Channel x Height x Width
    # 变量feature对应高和宽均为28像素的图像
    # 由于我们使用了transforms.ToTensor()，所以每个像素的数值为[0.0, 1.0]的32位浮点数。需要注意的是，feature的尺寸是 (C x H x W) 的


# 本函数已保存在d2lzh包中方便以后使用
def get_mnist_labels(labels):
    text_labels = ['0', '1', '2', '3', '4',
                   '5', '6', '7', '8', '9']
    return [text_labels[int(i)] for i in labels]


def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')


def show_mnist(images, labels):
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        # zip() 函数用于将可迭代的对象作为参数,
        # 将对象中对应的元素打包成一个个元组,
        # 然后返回由这些元组组成的列表
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# 输出一个batch(dataloader的数据)的图片和标签 8个一行 输出8行
def show_batch_mnist(dataloader):
    images, labels = next(iter(dataloader))
    img = utils.make_grid(images)
    # transpose 转置函数(x=0,y=1,z=2),新的x是原来的y轴大小，新的y是原来的z轴大小，新的z是原来的x大小
    # 相当于把x=1这个一道最后面去。
    img = img.numpy().transpose(1, 2, 0)
    for i in range(64):
        print(labels[i], end="")
        i += 1
        if i % 8 is 0:
            print(end='\n')
    plt.imshow(img)
    plt.savefig('one_batch_images')
    plt.show()

# 在测试集上评估准确率
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没有指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式，关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if('is_training' in net.__code__.co_varnames):  # 如果有is_training这个参数
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_num, n, batch_count, start, running_loss = 0.0, 0.0, 0, 0, time.time(), 0.0
        for i, data in enumerate(train_iter, 0):  # #0是下标起始位置默认为0
            # data 的格式[[inputs, labels]]
            X, y = data[0], data[1]
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            # 初始为0，清除上个batch的梯度信息
            optimizer.zero_grad()
            # 后向+优化
            l.backward()
            optimizer.step()

            # 记录训练过程中的train_loss
            train_l_sum += l.cpu().item()
            running_loss += l.cpu().item()
            if i % 100 == 99:
                print('[%d,%5d] loss :%.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
            train_loss.append(l)

            train_acc_num += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]  # y的size
            batch_count += 1

        # 记录训练过程中的train_accs和test_accs
        test_acc = evaluate_accuracy(test_iter, net)
        train_accs.append(train_acc_num/n)
        test_accs.append(test_acc)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' %
              (epoch+1, train_l_sum/batch_count, train_acc_num/n, test_acc, time.time()-start))

def draw_loss_process(title, iters, costs, label_cost, prefix):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("loss", fontsize=20)
    plt.plot(iters, costs, color='red', label=label_cost)
    plt.legend()
    plt.grid()
    plt.savefig(prefix + '_loss_result.png')
    plt.show()


def draw_acc_process(title, iters, costs, accs, label_cost, label_acc, prefix):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("acc", fontsize=20)
    plt.plot(iters, costs, color='red', label=label_cost)
    plt.plot(iters, accs, color='green', label=label_acc)
    plt.legend()
    plt.grid()
    plt.savefig(prefix + '_acc_result.png')
    plt.show()

if __name__ == '__main__':
    # 准备数据集
    mnist_train = datasets.MNIST(root='~/datasets/MNIST', train=True, download=False,
                                 transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root='~/datasets/MNIST', train=False, download=False,
                                transform=transforms.ToTensor())

    show_one_sample(0, train_dataset=mnist_train, test_dataset=mnist_test)

    # 展现一组图像
    X, y = [], []
    for i in range(10):
        X.append(mnist_train[i][0])
        y.append(mnist_train[i][1])
    show_mnist(X, get_mnist_labels(y))

    # 创建一个读取小批量数据样本的DataLoader实例
    batch_size = 64
    num_workers = 1
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    show_batch_mnist(dataloader = train_iter)

    # 和python中一样，类定义完之后实例化就很简单了，我们这里就实例化了一个net
    net = LeNet()
    print(net)

    lr, num_epochs = 0.001, 7
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 也可以选择SGD优化方法
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 绘制曲线需要
    train_loss = []
    train_accs = []
    test_accs = []

    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

    # 开始画图
    loss_iters = range(len(train_loss))
    acc_iters = range(len(train_accs))
    draw_loss_process('mnist_loss', loss_iters, train_loss, 'training_loss', 'mnist')
    draw_acc_process('mnist_acc', acc_iters, train_accs, test_accs, 'training_acc', 'test_acc', 'mnist')

    # 保存模型
    # 在PyTorch中，Module的可学习参数(即权重和偏差)，
    # 模块模型包含在参数中(通过model.parameters()访问)。
    # state_dict是一个从参数名称隐射到参数Tesnor的字典对象。
    PATH = "./mnist_net.pt"
    torch.save(net.state_dict(), PATH)  # 推荐的文件后缀名是pt或pth

    # 检验一个batch的分类情况
    images, labels = next(iter(test_iter))
    test_net = LeNet()
    test_net.load_state_dict(torch.load(PATH))
    # 输出的是每一类的对应概率，所以需要选择max来确定最终输出的类别
    # 测试集上面整体的准确率
    test_acc_num = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    # 进行评测的时候网络不更新梯度
    with torch.no_grad():
        for data in test_iter:
            images, labels = data
            outputs = test_net(images)
            test_acc_num += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.shape[0]
            for i in range(labels.shape[0]):
                label = labels[i]
                class_correct[label] += (outputs.argmax(dim=1)[i] == label).item()
                class_total[label] += 1
    print('Accuracy of the network on the test images: %d %%' % (100 * test_acc_num / total))
    # 10个类别的准确率
    for i in range(10):
        print('Accuracy of %d : %2d %%' % (
            i, 100 * class_correct[i] / class_total[i]))




