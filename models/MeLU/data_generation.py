import os
import torch
import random
import pickle
from tqdm import tqdm


def generate_train_data(dataset, config):

    # create dataset folder location
    dataset_path = "./models/MeLU/train_data"
    # if train data already exist then skip
    if os.path.exists(dataset_path):
        print('train data already exist')
        return 
    # else create dataset folder
    os.mkdir(dataset_path)

    # read all data
    items_data = dataset.get_clean_data()
    users_data = dataset.get_clean_users()
    ratings_data = dataset.get_clean_ratings()
    # hashmap for item information
    items_dict = {}
    for idx, row in items_data.iterrows():
        item_info = torch.tensor(row.array).long()
        items_dict[idx] = item_info.unsqueeze(0)
    # hashmap for users profile
    users_dict = {}
    for idx, row in users_data.iterrows():
        user_info = torch.tensor(row.array).long()
        users_dict[idx] = user_info.unsqueeze(0)

    # for every user that rated 8<=x<=100 items, create all trainig data
    idx = 0
    for _, u_id in tqdm(enumerate(users_dict.keys())):
        u_id = int(u_id)
        items_rated_df = ratings_data.loc[ratings_data['user_id']==u_id]
        items_x = items_rated_df['cellphone_id'].to_numpy()
        items_y = items_rated_df['rating'].to_numpy()
        items_len = len(items_x)
        indices = list(range(items_len))
        num_query = config['num_query']
        random.shuffle(indices)

        # create user support data. used for the local updates.
        support_x_app = None
        for i_id in items_x[indices[:num_query]]:
            i_id = int(i_id)
            items_x_converted = torch.cat((items_dict[i_id], users_dict[u_id]), 1)
            try:
                support_x_app = torch.cat((support_x_app, items_x_converted), 0)
            except:
                support_x_app = items_x_converted
        support_y_app = torch.FloatTensor(items_y[indices[:num_query]])

        # create user query data. used for the global updates.
        query_x_app = None
        for i_id in items_x[indices[num_query:]]:
            i_id = int(i_id)
            items_x_converted = torch.cat((items_dict[i_id], users_dict[u_id]), 1)
            try:
                query_x_app = torch.cat((query_x_app, items_x_converted), 0)
            except:
                query_x_app = items_x_converted
        query_y_app = torch.FloatTensor(items_y[indices[num_query:]])

        # save every user support and query in 4 files
        pickle.dump(support_x_app, open("{}/supp_x_{}.pkl".format(dataset_path, idx), "wb"))
        pickle.dump(support_y_app, open("{}/supp_y_{}.pkl".format(dataset_path, idx), "wb"))
        pickle.dump(query_x_app, open("{}/query_x_{}.pkl".format(dataset_path, idx), "wb"))
        pickle.dump(query_y_app, open("{}/query_y_{}.pkl".format(dataset_path, idx), "wb"))
        idx += 1
    
    # print number of train users
    training_set_size = int(len(os.listdir(dataset_path)) / 4)
    print("training set size: {}".format(training_set_size))