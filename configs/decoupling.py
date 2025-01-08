algorithm = 'Decoupling'
# dataset param
dataset = 'cifar-10'
input_channel = 3
num_classes = 10
root = '/data/yfli/CIFAR10'
noise_type = 'sym'
percent = 0.5
seed = 1
# model param
model1_type = 'resnet18'
model2_type = 'resnet18'
# train param
gpu = '1'
batch_size = 128
lr = 0.001
epochs = 200
num_workers = 4
exponent = 1
adjust_lr = 1
epoch_decay_start = 80
# result param
save_result = True