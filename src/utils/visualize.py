import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import os.path as osp
import torch
import seaborn as sns


def plot_dist(time_dist, save_path, name='time_dist'):

    fig, ax = plt.subplots()
    nbins = [0, 5, 10, 20, 60, 120, 1200]

    count = np.histogram(time_dist, bins=nbins)[0]
    count = count / int(count.sum())
    bar = ax.bar(list(range(1, 2*len(nbins)-1, 2)), count, width=1.5)
    plt.xticks(list(range(0, 2*len(nbins), 2)), list(map(str, nbins)))

    # Now we format the y-axis to display percentage
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.bar_label(bar, fmt="{:.0%}")
    plt.tight_layout()
    plt.savefig(osp.join(save_path, f'{name}.png'), dpi=150)


def plot_heatmap(raw_text, pred_similarity, target_sec, pred_sec, save_path, name):
    """
    pred_similarity: (S, T)
    """
    S, T = pred_similarity.shape[0], pred_similarity.shape[1]
    target_sec, pred_sec = target_sec[:S], pred_sec[:S] # filter padding target time (S,)
    low, high = torch.floor(target_sec[:, None]), torch.ceil(target_sec[:, None]) #(S, T)
    timestamp = torch.arange(T, device=target_sec.device)[None, :] #(1, T)
    
    target_similarity = (timestamp >= (low)) * (timestamp <= (high)) # True for correct time (S, T)

    # Set up the matplotlib figure
    f, ax = plt.subplots(2, 1)
    
    pred_similarity = pred_similarity.detach().cpu().numpy()
    target_similarity = target_similarity.detach().cpu().numpy()
    tgt_time = np.round(target_sec.detach().cpu().numpy())
    pred_time = np.round(pred_sec.detach().cpu().numpy())
    sim_time = pred_similarity.argmax(-1)

    pred_error = pred_time - tgt_time
    sim_error = sim_time - tgt_time
    regression_error = np.abs(pred_error).mean()
    similarity_error = np.abs(sim_error).mean()

    target_tick = [i.strip()[3:] + ' {: >3.0f}'.format(j) for i, j in zip(raw_text, tgt_time)]
    pred_tick = [i.strip()[3:] + ' {: >3.0f}|{: >3.0f}'.format(j, k) for i, j, k in zip(raw_text, pred_error, sim_error)]

    sns.heatmap(target_similarity, cmap='viridis', vmax=1, vmin=0, ax=ax[0], xticklabels=30, yticklabels=target_tick, cbar=False)
    sns.heatmap(pred_similarity, cmap='viridis', vmax=1, vmin=0, ax=ax[1], xticklabels=30, yticklabels=pred_tick, cbar=False)
    
    ax[0].tick_params(axis='x', labelsize=6)
    ax[1].tick_params(axis='x', labelsize=6)
    ax[0].tick_params(axis='y', labelsize=5)
    ax[1].tick_params(axis='y', labelsize=5)
    ax[0].set_title('Target', fontsize=7)
    ax[1].set_title('Pred Error {:.0f}|{:.0f}'.format(regression_error, similarity_error), fontsize=7)

    plt.tight_layout()
    plt.savefig(osp.join(save_path, f'{name}.png'), dpi=400)


def plot_heatmap_htmalign(raw_text, pred_similarity, target_start, target_end, tgt_sign, pred_sec, save_path, name):

    S, T = pred_similarity.shape[0], pred_similarity.shape[1]
    target_start, target_end, tgt_sign, pred_sec = target_start[:S], target_end[:S], tgt_sign[:S], pred_sec[:S] # filter padding (S,)

    low, high = torch.floor(target_start[:, None]), torch.ceil(target_end[:, None]) #(S, T)
    timestamp = torch.arange(T, device=pred_sec.device)[None, :] #(1, T)
    target_similarity = (timestamp >= (low)) * (timestamp <= (high)) # True for correct time (S, T)

    # filter unalignable and drop some alignable
    tgt_sign = (tgt_sign == 1)
    target_similarity = target_similarity[tgt_sign]
    pred_similarity = pred_similarity[tgt_sign]
    target_start, target_end, pred_sec = target_start[tgt_sign], target_end[tgt_sign], pred_sec[tgt_sign]
    alignable_text = [text for text, sign in zip(raw_text, tgt_sign) if sign]

    # Set up the matplotlib figure
    f, ax = plt.subplots(2, 1)
    
    pred_similarity = pred_similarity.detach().cpu().numpy()
    target_similarity = target_similarity.detach().cpu().numpy()
    tgt_start = np.round(target_start.detach().cpu().numpy())
    tgt_end = np.round(target_end.detach().cpu().numpy())
    pred_time = np.round(pred_sec.detach().cpu().numpy())
    sim_time = pred_similarity.argmax(-1)
    pred_recall = np.mean((tgt_start <= pred_time) & (pred_time<= tgt_end))
    sim_recall = np.mean((tgt_start <= sim_time) & (sim_time<= tgt_end))

    target_tick = [i.strip()[:30] + ' [{: >3.0f}, {: >3.0f}]'.format(j, k) for i, j, k in zip(alignable_text, tgt_start, tgt_end)]
    pred_tick = [i.strip()[:30] + ' {: >3.0f}|{: >3.0f}'.format(j, k) for i, j, k in zip(alignable_text, pred_time, sim_time)]

    sns.heatmap(target_similarity, cmap='viridis', vmax=1, vmin=0, ax=ax[0], xticklabels=30, yticklabels=target_tick, cbar=False)
    sns.heatmap(pred_similarity, cmap='viridis', vmax=1, vmin=0, ax=ax[1], xticklabels=30, yticklabels=pred_tick, cbar=False)
    
    ax[0].tick_params(axis='x', labelsize=6)
    ax[1].tick_params(axis='x', labelsize=6)
    ax[0].tick_params(axis='y', labelsize=5)
    ax[1].tick_params(axis='y', labelsize=5)
    ax[0].set_title('Target', fontsize=7)
    ax[1].set_title('Pred (recall={:.2f}|{:.2f})'.format(pred_recall, sim_recall), fontsize=7)

    plt.tight_layout()
    plt.savefig(osp.join(save_path, f'{name}.png'), dpi=400)


def plot_sim_heatmap(raw_text, pred_similarity, time_mask, save_path, name):
    """
    pred_similarity: (S, T)
    """
    S, T = pred_similarity.shape[0], pred_similarity.shape[1]
    time_mask = time_mask[:S, :T] # True for correct time (S, T)
    pred_sec = pred_similarity.argmax(-1) # (S,)

    # Set up the matplotlib figure
    f, ax = plt.subplots(2, 1)
    
    correct_list = torch.gather(time_mask, 1, pred_sec[:, None]) == True
    recall = torch.mean(correct_list.float()) #(S,)
    pred_similarity = pred_similarity.detach().cpu().numpy()
    time_mask = time_mask.detach().cpu().numpy()

    target_tick = [i.strip()[3:] for i in raw_text]

    pred_tick = []
    for i, j, k in zip(raw_text, pred_sec, correct_list):
        if k:
            pred_tick.append(i.strip()[3:] + ' {: >3.0f} \u2714'.format(j))
        else:
            pred_tick.append(i.strip()[3:] + ' {: >3.0f} \u2718'.format(j))

    sns.heatmap(time_mask, cmap='viridis', vmax=1, vmin=0, ax=ax[0], xticklabels=30, yticklabels=target_tick, cbar=False)
    sns.heatmap(pred_similarity, cmap='viridis', vmax=1, vmin=0, ax=ax[1], xticklabels=30, yticklabels=pred_tick, cbar=False)
    
    ax[0].tick_params(axis='x', labelsize=6)
    ax[1].tick_params(axis='x', labelsize=6)
    ax[0].tick_params(axis='y', labelsize=5)
    ax[1].tick_params(axis='y', labelsize=5)
    ax[0].set_title('Target', fontsize=7)
    ax[1].set_title('Pred recall={:.0%}'.format(recall), fontsize=7)

    plt.tight_layout()
    plt.savefig(osp.join(save_path, f'{name}.png'), dpi=400)


def plot_sim_heatmap_with_sign(raw_text, pred_similarity, sign, time_mask, save_path, name):
    """
    pred_similarity: (S, T)
    """
    S, T = pred_similarity.shape[0], pred_similarity.shape[1]
    time_mask = time_mask[:S, :T] # True for correct time (S, T)
    sign = sign[:S]
    pred_sec = pred_similarity.argmax(-1) # (S,)

    num_texts = S #min(S, int(T / 60) * 4)
    display_mask = (torch.rand(S) < (num_texts / S)) * (sign == 1).cpu()

    # Set up the matplotlib figure
    f, ax = plt.subplots(2, 1)
    
    correct_list = torch.gather(time_mask, 1, pred_sec[:, None]) == True
    # only compute recall of alignable texts
    recall = torch.mean(correct_list[sign == 1].float()) #(S,)

    pred_similarity = pred_similarity.detach().cpu().numpy()
    time_mask = time_mask.detach().cpu().numpy()

    pred_tick, target_tick = [], []
    for i, j, m, n, k in zip(raw_text, pred_sec, correct_list, sign, display_mask):
        if not k:
            continue

        text = i.strip()[:50]
        if not n:
            pred_tick.append(text + ' {: >3.0f} \u25E6'.format(j))
            target_tick.append(text + ' \u25E6'.format(j))
        elif m:
            pred_tick.append(text + ' {: >3.0f} \u2714'.format(j))
            target_tick.append(text + '  ')
        else:
            pred_tick.append(text + ' {: >3.0f} \u2718'.format(j))
            target_tick.append(text + '  ')

    sns.heatmap(time_mask[display_mask], cmap='viridis', vmax=1, vmin=0, ax=ax[0], xticklabels=30, yticklabels=target_tick, cbar=False)
    sns.heatmap(pred_similarity[display_mask], cmap='viridis', vmax=1, vmin=0, ax=ax[1], xticklabels=30, yticklabels=pred_tick, cbar=False)
    
    ax[0].tick_params(axis='x', labelsize=6)
    ax[1].tick_params(axis='x', labelsize=6)
    ax[0].tick_params(axis='y', labelsize=5)
    ax[1].tick_params(axis='y', labelsize=5)
    ax[0].set_title('Target', fontsize=7)
    ax[1].set_title('Pred recall={:.0%}'.format(recall), fontsize=7)

    plt.tight_layout()
    plt.savefig(osp.join(save_path, f'{name}.png'), dpi=400)