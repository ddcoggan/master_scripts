import matplotlib.pyplot as plt

def plot_performance(performance, metrics, outdir):

    fig, axes = plt.subplots(1, len(metrics),
                                     figsize=(len(metrics * 4), 4))
    train_vals = performance[performance['train_eval'] == 'train']
    eval_vals = performance[performance['train_eval'] == 'eval']
    for m, metric in enumerate(metrics):
        ax = axes if len(metrics) == 1 else axes[m]
        train_x, train_y = (train_vals['epoch'].values,
                            train_vals[metric].values)
        eval_x, eval_y = (eval_vals['epoch'].values,
                          eval_vals[metric].values)
        ax.plot(train_x, train_y, label='train')
        ax.plot(eval_x, eval_y, label='eval')
        ax.set_xlabel('epoch')
        ax.set_ylabel(metric)
        ax.grid()
        if metric == 'loss_contr':
            ax.set_yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{outdir}/performance.png')
    plt.close()

