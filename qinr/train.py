import torch
from torch.utils.data import DataLoader
import numpy as np
from modules import Hybridren, FFN, Siren, Siren_bn
from image.data_process import ImageFitting
from sound.data_process import AudioFile
from config_parser import ConfigArgumentParser, save_args
from Log import Logger
import sys
import os
import time
from diff_operators import gradient, laplace
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

parser = ConfigArgumentParser()
parser.add_argument('--type', type=str, default='sound')
parser.add_argument('--epochs', type=int, default=301)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epoch_til_summary', type=int, default=1)
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--lr_scheduler', type=str, default='cosine')
parser.add_argument('--criterion', type=str, default='mse')
parser.add_argument('--use_cuda', type=bool, default=0)
parser.add_argument('--learning_rate', type=float, default=0.03)

parser.add_argument('--model', type=str, default='hybridren')
parser.add_argument('--in_features', type=int, default=1)
parser.add_argument('--hidden_features', type=int, default=8)
parser.add_argument('--hidden_layers', type=int, default=2)
parser.add_argument('--first_omega_0', type=float, default=1)
parser.add_argument('--hidden_omega_0', type=float, default=1)
parser.add_argument('--spectrum_layer', type=int, default=2)
parser.add_argument('--use_noise', type=float, default=0)

args = parser.parse_args()


# --config ./image/config/siren.yaml
# save_args(args, './' + args.type + '/config/' + args.model + '.yaml')


class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_val(self, step):
        assert 0 <= step <= self.total_steps

        if step < self.warmup_steps:
            return self.max_val * (1 + np.cos(step * np.pi / (2 * self.total_steps))) / 2
        elif self.warmup_steps <= step <= self.total_steps:
            return max(self.max_val / 10, self.max_val * (1 + np.cos(step * np.pi / self.total_steps)) / 2)
        else:
            raise AssertionError


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, total_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = total_steps

    def get_val(self, iter) -> float:
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)


class Trainer(object):
    def __init__(self, model, criterion, optimizer, dataloader, epochs, lr_scheduler, epoch_til_summary,
                 use_cuda=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.use_cuda = use_cuda
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.epoch_til_summary = epoch_til_summary

        if self.use_cuda:
            self.model = self.model.cuda()

    def run(self):
        for epoch in range(self.epochs):
            loss = self.train(epoch)
            if not epoch % self.epoch_til_summary:
                print("epoch %d, Total loss %0.6f" % (epoch, loss))
                path = "./checkpoint/" + str(epoch) + ".pth"
                torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def train(self, epoch):
        self.model.train()
        total_loss = 0
        for step, (x, y) in enumerate(self.dataloader):
            if self.use_cuda:
                x = x.cuda()
                y = y.cuda()
            out, coords = self.model(x)
            loss = self.criterion(out, y)
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            # self.optimizer.param_groups[0]['lr'] = lr_scheduler.get_val(step=epoch)
            self.optimizer.step()
        return total_loss

    def evaluation(self):
        self.model.eval()
        with torch.no_grad():
            eval_loss = 0
            eval_acc = 0
            for i, (x, y) in enumerate(self.dataset, self.iteration + 1):
                if self.use_cuda:
                    x = x.cuda()
                    y = y.cuda()
                out = self.model(x)
                loss = self.criterion(out, y)
                eval_loss += loss.item()
                prediction = torch.max(out, 1)[1]
                pred_correct = (prediction == y).sum()
                eval_acc += pred_correct.item()
            print('evaluation loss : {:.6f}, acc : {:.6f}'.format(eval_loss / len(val_data), eval_acc / len(val_data)))


def initial_log():
    log_path = './Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    sys.stdout = Logger(log_file_name)
    sys.stderr = Logger(log_file_name)


if __name__ == "__main__":
    initial_log()
    print(args)

    if args.type == 'image':
        dataloader = DataLoader(ImageFitting(args.img_size), batch_size=args.batch_size, pin_memory=True)
    elif args.type in ['sound', 'wave']:
        dataloader = DataLoader(AudioFile('./sound/gt_bach.wav'), batch_size=args.batch_size, pin_memory=True)

    if args.model == 'siren_bn':
        model = Siren_bn(in_features=args.in_features, out_features=1, hidden_features=args.hidden_features,
                         hidden_layers=args.hidden_layers, first_omega_0=args.first_omega_0,
                         hidden_omega_0=args.hidden_omega_0, activ='sine')
    elif args.model == 'relu':
        model = Siren_bn(in_features=args.in_features, out_features=1, hidden_features=args.hidden_features,
                         hidden_layers=args.hidden_layers, first_omega_0=args.first_omega_0,
                         hidden_omega_0=args.hidden_omega_0, activ='relu')
    elif args.model == 'tanh':
        model = Siren_bn(in_features=args.in_features, out_features=1, hidden_features=args.hidden_features,
                         hidden_layers=args.hidden_layers, first_omega_0=args.first_omega_0,
                         hidden_omega_0=args.hidden_omega_0, activ='tanh')
    elif args.model == 'siren':
        model = Siren(in_features=args.in_features, out_features=1, hidden_features=args.hidden_features,
                      hidden_layers=args.hidden_layers, first_omega_0=args.first_omega_0,
                      hidden_omega_0=args.hidden_omega_0)
    elif args.model == 'ffn':
        model = FFN(in_features=args.in_features, out_features=1, hidden_features=args.hidden_features,
                    hidden_layers=args.hidden_layers)
    elif args.model == 'hybridren':
        model = Hybridren(in_features=args.in_features, out_features=1, hidden_features=args.hidden_features,
                          hidden_layers=args.hidden_layers, spectrum_layer=args.spectrum_layer,
                          use_noise=args.use_noise)
    elif args.model == 'relu+rff':
        model = Siren_bn(in_features=args.in_features, out_features=1, hidden_features=args.hidden_features,
                         hidden_layers=args.hidden_layers, first_omega_0=args.first_omega_0,
                         hidden_omega_0=args.hidden_omega_0, activ='relu', rff=True)

    print(model)
    print('parameters:%d' % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if args.use_cuda:
        model.cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD((param for param in model.parameters() if param.requires_grad),
                                    lr=args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad),
                                     lr=args.learning_rate)

    if args.lr_scheduler == 'cosine':
        lr_scheduler = CosineDecayScheduler(max_val=args.learning_rate, warmup_steps=0, total_steps=args.epochs)
    elif args.lr_scheduler == 'linear':
        lr_scheduler = LinearDecaySchedule(start_val=args.learning_rate, final_val=0, total_steps=args.epochs)

    if args.criterion == 'mse':
        criterion = lambda x, y: ((x - y) ** 2).mean()

    trainer = Trainer(model, criterion, optimizer, dataloader, args.epochs, lr_scheduler, args.epoch_til_summary,
                      use_cuda=args.use_cuda)
    trainer.run()

    # 可视化
    # checkpoint = torch.load("./sound/checkpoint/siren.pth")
    # model.load_state_dict(checkpoint['model_state_dict'])
    model_input, ground_truth = next(iter(dataloader))
    if args.use_cuda:
        model_input = model_input.cuda()
    model_output, coords = trainer.model(model_input)
    # torch.save(model_output, './tanh_output.pt')
    if args.type == 'image':
        # img_grad = gradient(model_output, coords)
        # img_laplacian = laplace(model_output, coords)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=600)
        axes[0].imshow(model_output.view(args.img_size, args.img_size).detach().numpy(), cmap='gray')
        # axes[1].imshow(img_grad.norm(dim=-1).view(args.img_size, args.img_size).detach().numpy())
        # axes[2].imshow(img_laplacian.view(args.img_size, args.img_size).detach().numpy())
    elif args.type == 'sound':
        fig, axes = plt.subplots(1, 2, dpi=600)
        axes[0].plot(coords.squeeze().detach().cpu().numpy(), model_output.squeeze().detach().cpu().numpy())
        axes[1].plot(coords.squeeze().detach().cpu().numpy(), ground_truth.squeeze().detach().numpy())
    plt.show()
    # wavfile.write('./wave/output.wav', AudioFile('./wave/gt_bach.wav').rate,model_output.squeeze().detach().cpu().numpy())
