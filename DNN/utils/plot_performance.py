import matplotlib.pyplot as plt
import numpy as np

def plot_performance(performance, metrics, outdir):

    if len(metrics) < 3:
        nrows, ncols = 1, len(metrics) + 1
    else:
        nrows, ncols = 2, int(np.ceil((len(metrics)+1)/2))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(ncols * 4, nrows * 4))
    train_vals = performance[performance['train_eval'] == 'train']
    eval_vals = performance[performance['train_eval'] == 'eval']
    for m, metric in enumerate(metrics + ['lr']):
        ax = axes.flatten()[m]
        train_x, train_y = (train_vals['epoch'].values,
                            train_vals[metric].values)
        eval_x, eval_y = (eval_vals['epoch'].values,
                          eval_vals[metric].values)
        ax.plot(train_x, train_y, label='train')
        ax.plot(eval_x, eval_y, label='eval')
        ax.set_xlabel('epoch')
        ax.set_ylabel(metric)
        ax.grid()
        if metric.startswith('l'):
            ax.set_yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{outdir}/performance.png')
    plt.close()

