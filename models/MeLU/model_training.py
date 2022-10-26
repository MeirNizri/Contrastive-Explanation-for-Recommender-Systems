import os
import torch
import pickle
import random
from tqdm import tqdm


def training(melu, model_save=True):
    # create model folder location
    model_filename = "./models/MeLU/model_info.pkl"
    train_data_path = "./models/MeLU/train_data"
    # if model already exist then skip
    if os.path.exists(model_filename):
        trained_state_dict = torch.load(model_filename)
        melu.load_state_dict(trained_state_dict)
        print('trained model already exist')
        return 

    # Load training dataset.
    training_set_size = int(len(os.listdir(train_data_path)) / 4)
    supp_xs_s = []
    supp_ys_s = []
    query_xs_s = []
    query_ys_s = []
    for idx in tqdm(range(training_set_size)):
        supp_xs_s.append(pickle.load(open("{}/supp_x_{}.pkl".format(train_data_path, idx), "rb")))
        supp_ys_s.append(pickle.load(open("{}/supp_y_{}.pkl".format(train_data_path, idx), "rb")))
        query_xs_s.append(pickle.load(open("{}/query_x_{}.pkl".format(train_data_path, idx), "rb")))
        query_ys_s.append(pickle.load(open("{}/query_y_{}.pkl".format(train_data_path, idx), "rb")))
    total_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
    del(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)

    # check if to run on GPU
    config = melu.get_config()
    if config['use_cuda']:
        melu.cuda()
    
    # set train details
    num_epoch = config['num_epoch']
    batch_size = config['batch_size']
    num_local_updates = config['inner']
    training_set_size = len(total_dataset)
    melu.train()

    
    for _ in tqdm(range(num_epoch)):
        # split the train data to batches
        random.shuffle(total_dataset)
        num_batch = int(training_set_size / batch_size)
        a,b,c,d = zip(*total_dataset)
        for i in range(num_batch):
            try:
                supp_xs = list(a[batch_size*i:batch_size*(i+1)])
                supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                query_xs = list(c[batch_size*i:batch_size*(i+1)])
                query_ys = list(d[batch_size*i:batch_size*(i+1)])
            except IndexError:
                continue
            # global update for every batch
            melu.global_update(supp_xs, supp_ys, query_xs, query_ys, num_local_updates)
    
    # save model parameters
    if model_save:
        torch.save(melu.state_dict(), model_filename)
    