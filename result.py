import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def result_plt(model_name, model_idx, train_record, validbest_record):
    print('Plotting All Record')
    site_list = ['nyu', 'peking', 'ohsu', 'kki', 'ni']
    colors = ['green','blue','red','yellow','orange']
    metrics = ['loss','accuracy','auc']
    val_metrics = ['val_loss','val_accuracy','val_auc']

    root_dir = './image/' + model_name +'/'
    branch_dir = root_dir + model_idx
    os.makedirs(branch_dir, exist_ok=True)

    # training record
    for metric in metrics:
        fig1, ax1 = plt.subplots()

        for i, site in enumerate(site_list):
            site_train_record = train_record[i]

            length = np.arange(len(site_train_record[metric]))
            ax1.plot(length, site_train_record[metric], color=colors[i], marker='o', linestyle='dashed', linewidth=.5, markersize=.5, label=site)
            ax1.set_xticks(np.arange(0,len(site_train_record['loss']),5))

        plt.legend()
        ax1.set_ylabel(metric.title())
        ax1.set_xlabel('Epochs')
        ax1.set_title('Site Training '+metric.title())
        plt.savefig(branch_dir +'/trainset_'+metric+ '.png')

        plt.clf()

    # validation record
    for metric in val_metrics:
        fig1, ax1 = plt.subplots()

        for i, site in enumerate(site_list):
            site_train_record = train_record[i]

            length = np.arange(len(site_train_record[metric]))
            ax1.plot(length, site_train_record[metric], color=colors[i], marker='o', linestyle='dashed', linewidth=.5, markersize=.5, label=site)
            ax1.set_xticks(np.arange(0,len(site_train_record['loss']),5))

        plt.legend()
        ax1.set_ylabel(metric.title())
        ax1.set_xlabel('Epochs')
        ax1.set_title('Site Validation '+metric.title())
        plt.savefig(branch_dir +'/'+metric+ '.png')
        plt.clf()

    # validbest_record
    for metric_idx, metric in enumerate(metrics):
        fig1, ax1 = plt.subplots()
        x_list = list()
        for i, site in enumerate(site_list):
            site_test_record = validbest_record[i]
            x_list.append(site_test_record[metric_idx])

        ax1.plot(site_list, x_list, color=colors[0], marker='o', linestyle='dashed', linewidth=.5, markersize=.5)

        ax1.set_ylabel(metric.title())
        ax1.set_xlabel('Sites')
        ax1.set_title('Site Best Validation '+metric.title())
        plt.savefig(branch_dir +'/bestvalid_'+metric+'.png')
        plt.clf()
        print('Validation Best',metric,np.mean(x_list), x_list)


if __name__ == "__main__":
    model_name = 'roi_test'

    with open('./results/' + model_name + '.pickle', 'rb') as f:
        train_record = pickle.load(f)
        validbest_record = pickle.load(f)

    result_plt(model_name, train_record, validbest_record)
