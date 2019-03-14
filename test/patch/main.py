from torch.autograd import Variable
from torchsummary import summary
from model import *
from data import *
from plot import *
#import matplotlib.pyplot as plt
#import numpy
#import torchvision


def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (1 - decay_rate)


torch.backends.cudnn.benchmark = True

model = cnn_model(3, 2)
model = model.cuda()
# summary(model, (3, 96, 96))

[train_loader, train_dataset]= loadtraindata()
[test_loader, test_dataset]= loadtestdata()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

#img = torchvision.utils.make_grid(train_dataset[10][0]).numpy()
#plt.imshow(numpy.transpose(img,(1,2,0)))
#plt.show()

train_acc = [0 for i in range(num_epoches)]
test_acc = [0 for i in range(num_epoches)]

for epoch in range(num_epoches):
    print('epoch {}'.format(epoch+1))
    print('*'*10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, num_epoches, running_loss / (batch_size * i),
            running_acc / (batch_size * i)))
            a = (i*100/(len(train_loader)))
            print("%.2f" % a, '%')
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))
    train_acc[epoch] = running_acc / (len(train_dataset))
    model.eval().cuda()
    eval_loss = 0.0
    eval_acc = 0.0
    for data in test_loader:
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    test_acc[epoch] = eval_acc / (len(test_dataset))
    adjust_learning_rate(optimizer, decay_rate)

acc_plot(train_acc, test_acc)