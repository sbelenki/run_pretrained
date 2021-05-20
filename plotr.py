import datetime
import time
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

DIR_CHKPT = '/mnt/d/fp/checkpoints/'
DIR_CSV = '/mnt/d/fp/csv/'


def loss_from_log(filename, phase='test'):
    pt = 'Phase Type: ' + phase
    vals = []
    with open(filename) as log_in:
        for line in log_in.readlines():
            if pt in line:
                m = re.search('loss: ([\\d\\.]+)?', line)
                if m:
                    vals.append(float(m.group(1)))

    print(f'{filename} - {len(vals)} entries; max={np.max(vals)}; min={np.min(vals)}')

    return vals


def load_csv_col(filename, col, limit=None):
    vals = []
    line = 0
    with open(filename) as csv_in:
        for row in csv.reader(csv_in, delimiter=','):
            line += 1
            if line == 1:
                continue
            vals.append(float(row[col]))

    if limit:
        vals = vals[:limit]

    print(f'{filename} - {len(vals)} entries; max={np.max(vals)}; min={np.min(vals)}')
    return vals


def plot(entries, title='TITLE', window_length=0,
         log_x=False, log_y=False,
         xlabel='Epoch', ylabel=None,
         xlim=None, ylim=None,
         legend_loc=None, tight=False):
    ts = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S')

    fig = plt.figure(figsize=(7, 5))

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')

    for entry in entries:
        data = entry['data']
        if window_length:
            data = savgol_filter(data, window_length, 1)

        args = {}
        if 'marker' in entry:
            args['marker'] = entry['marker']
        if 'color' in entry:
            args['color'] = entry['color']

        plt.plot(data, label=entry['label'], **args)

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    plt.grid(linestyle=':')
    plt.title(title)
    if legend_loc:
        plt.legend(loc=legend_loc)
    else:
        plt.legend()

    if tight:
        plt.tight_layout()

    plt.savefig(f'{ts} {title}.png')
    plt.show()


def plot_loss():
    entries = [
        {
            'data': load_csv_col(DIR_CSV + 'baseline_50eps_loss.csv', 2),
            'label': 'baseline'
        },
        {
            'data': load_csv_col(DIR_CSV + 'swav_FT_10eps_loss.csv', 2),
            'label': 'swav'
        },
        {
            'data': load_csv_col(DIR_CSV + 'rotnet_FT_10eps_loss.csv', 2),
            'label': 'rotnet'
        },
        {
            'data': load_csv_col(DIR_CSV + 'simclr_FT_5ep_loss.csv', 2),
            'label': 'simclr'
        }
    ]
    plot(entries, title='loss', window_length=51)


def plot_ssim():
    entries = [
        {
            'data': load_csv_col(DIR_CSV + 'baseline_50eps_SSIM.csv', 2, limit=10),
            'label': 'Baseline',
            'marker': 'o',
            'color': 'black'
        },
        {
            'data': load_csv_col(DIR_CSV + 'swav_FT_10eps_SSIM.csv', 2),
            'label': 'SwAV',
            'marker': '^'
        },
        {
            'data': load_csv_col(DIR_CSV + 'rotnet_FT_10eps_SSIM.csv', 2),
            'label': 'RotNet',
            'marker': 'v'
        },
        {
            'data': load_csv_col(DIR_CSV + 'simclr_FT_10eps_ssim.csv', 2),
            'label': 'SimCLR',
            'marker': 'x'
        },
        {
            'data': load_csv_col(DIR_CSV + 'pirl_FT_10ep_(PT_1000ep)_ssim.csv', 2),
            'label': 'PIRL',
            'marker': '+'
        }
    ]
    plot(entries, title='Fine-tuning SSIM Per Epoch',
         xlabel='Epoch', ylabel='Structural Similarity (SSIM)',
         window_length=3, ylim=[0.63, 0.74], legend_loc=3)


def plot_pt_loss():
    entries = [
        {
            'data': load_csv_col(DIR_CSV + 'swav_PT_30eps_train_loss.csv', 2),
            'label': 'SwAV',
            'marker': '^'
        },
        {
            'data': load_csv_col(DIR_CSV + 'rotnet_PT_100eps_Training_Loss.csv', 2),
            'label': 'RotNet',
            'marker': 'v'
        },
        {
            'data': load_csv_col(DIR_CSV + 'simclr_PT_200ep_Training_Loss.csv', 2),
            'label': 'SimCLR',
            'marker': 'x'
        },
        {
            'data': np.array(load_csv_col(DIR_CSV + 'pirl_PT_1000ep-Training_Loss.csv', 2)) / 160.0,
            'label': 'PIRL',
            'marker': '+'
        }
    ]
    plot(entries, title='Pre-training Loss Per Epoch', window_length=11, xlim=[0, 60], ylabel='Loss')


def main():
    #plot_loss()
    #plot_ssim()
    plot_pt_loss()


if __name__ == '__main__':
    main()
