import sys
from ModelWrapper import ModelWrapper
import torch
from torchvision import transforms, datasets
import resnet
import numpy as np
import random
import os


data_name = 'cifar10'
model_name = 'resnet'

lr = 1e-4
train_batch_size = 128
train_epoch = 1001
eval_batch_size = 256
# width
k = 64
# 0 to 0.2
label_noise = float(sys.argv[1])
# 1 or 10
num_classes = int(sys.argv[2])
only_read_out = False
# 0.1 to 8
lambda_log = float(sys.argv[3])
lambda_ = 10 ** (-lambda_log)
seed = int(sys.argv[4])

save_indivs = int(sys.argv[5]) == 1
save_epochs = int(sys.argv[6]) == 1
compute_ntk = int(sys.argv[7]) == 1

exp_name = str(sys.argv[8])
reg = float(sys.argv[9])

print('num_classes: ' + str(num_classes) + '\n' +
      'noise: ' + str(label_noise) + '\n' +
      'lambda_log: ' + str(lambda_log) + '\n' +
      'seed: ' + str(seed) + '\n' +
      'save_indivs: ' + str(save_indivs) + '\n' +
      'save_epochs: ' + str(save_epochs) + '\n' +
      'compute_ntk: ' + str(compute_ntk) + '\n' +
      'exp_name: ' + exp_name + '\n')

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.benchmark = False#
# torch.backends.cudnn.deterministic = True#

# 64 0.15 1 10 0

dataset = datasets.CIFAR10
train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])

eval_transform = transforms.Compose([transforms.ToTensor()])
model = resnet.resnet18(k, num_classes)

# load data
train_data = dataset('datasets', train=True, transform=train_transform, download=True)
train_targets = np.array(train_data.targets)
data_size = len(train_targets)
random_index = random.sample(range(data_size), int(data_size*label_noise))
random_part = train_targets[random_index]
np.random.shuffle(random_part)
train_targets[random_index] = random_part
train_data.targets = train_targets.tolist()

noise_data = dataset('datasets', train=True, transform=train_transform, download=True)
noise_data.targets = random_part.tolist()
noise_data.data = train_data.data[random_index]


test_data = dataset('datasets', train=False, transform=eval_transform, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=0,
                                           drop_last=False)
noise_loader = torch.utils.data.DataLoader(noise_data, batch_size=train_batch_size, shuffle=False, num_workers=0,
                                           drop_last=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, num_workers=0,
                                          drop_last=False)

# build model
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = model.to(device)


criterion = torch.nn.CrossEntropyLoss()
if only_read_out:
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr, weight_decay=lambda_)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_)

wrapper = ModelWrapper(model, optimizer, criterion, device, only_read_out, reg)

# train the model
save_path = '/home/<use>/scratch/DD/' + exp_name
if not os.path.exists(save_path):
    os.makedirs(save_path)

wrapper.train()

all_indivs = []
# train and test error
logs = np.zeros((3, train_epoch))
for id_epoch in range(train_epoch):
    # train loop

    avg_train_acc = []
    for id_batch, (inputs, targets) in enumerate(train_loader):

        loss, acc, _, Q = wrapper.train_on_batch(inputs, targets)
        avg_train_acc += [acc]
        print("epoch:{}/{}, batch:{}/{}, loss={}, err={}".
              format(id_epoch + 1, train_epoch, id_batch + 1, len(train_loader), loss, 1.0 - acc))

    wrapper.eval()
    test_loss, test_acc, indivs = wrapper.eval_all(test_loader)
    all_indivs += [indivs.data.cpu().numpy()[None]]

    logs[0, id_epoch] = 1.0 - np.mean(avg_train_acc)
    logs[1, id_epoch] = 1.0 - test_acc
    logs[2, id_epoch] = Q

    print("epoch:{}/{}, batch:{}/{}, testing...".format(id_epoch + 1, train_epoch, id_batch + 1, len(train_loader)))
    print("clean: loss={}, err={}".format(test_loss, 1.0 - test_acc))

    if save_epochs:
        torch.save(model.state_dict(),
                   (save_path +
                    '/ckpt_num_classes_' + str(num_classes) +
                    '_noise_' + str(label_noise) +
                    '_lambda_log_' + str(lambda_log) +
                    '_seed_' + str(seed) +
                    '_epoch_' + str(id_epoch) + '.pkl'))

    wrapper.train()

if save_indivs:
    all_indivs = np.concatenate(all_indivs, 0)
    np.save((save_path +
             '/indivs_num_classes_' + str(num_classes) +
             '_noise_' + str(label_noise) +
             '_lambda_log_' + str(lambda_log) +
             '_seed_' + str(seed)), all_indivs)

np.save((save_path +
         '/log_num_classes_' + str(num_classes) +
         '_noise_' + str(label_noise) +
         '_lambda_log_' + str(lambda_log) +
         '_seed_' + str(seed)), logs)
