import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict

from models.MeLU.embeddings import item, user
from models.MeLU.data_generation import generate_train_data


class user_preference_estimator(torch.nn.Module):
    def __init__(self, config, dataset, embeddings):
        super(user_preference_estimator, self).__init__()   
        # get clean data and number of features
        self.items_data = dataset.get_clean_data()
        self.users_data = dataset.get_clean_users()
        self.item_num_cols = self.items_data.shape[1]
        self.user_num_cols = self.users_data.shape[1]
        # item and user embedding
        self.item_emb = embeddings.item(config, self.items_data)
        self.user_emb = embeddings.user(config, self.users_data) 
        torch.autograd.set_detect_anomaly(True)      
        # set model structure
        self.embedding_dim = config['embedding_dim']
        self.fc1_in_dim = config['embedding_dim'] * config['num_features']
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        self.use_cuda = config['use_cuda']
        self.fc1 = torch.nn.Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = torch.nn.Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = torch.nn.Linear(self.fc2_out_dim, 1)

    def forward(self, x, training = True):
        # find embedding of x
        # split between item and user columns
        items_col = x[:, :self.item_num_cols]
        users_col = x[:, self.item_num_cols:]
        # get embedding for every (item,user) pair in x
        item_emb = self.item_emb(items_col)
        user_emb = self.user_emb(users_col)
        # apply neural network
        x = torch.cat((item_emb, user_emb), 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.linear_out(x)
        return x



class MeLU(torch.nn.Module):
    def __init__(self, dataset, config, embeddings):
        super(MeLU, self).__init__()
        # create train data
        self.dataset = dataset
        self.config = config
        # generate_train_data(dataset, config)
        self.last_user = None
        # set local updates details
        self.use_cuda = config['use_cuda']
        self.local_lr = config['local_lr']
        self.model = user_preference_estimator(config, dataset, embeddings)
        # store parameters for global updates
        self.store_parameters()
        self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'linear_out.weight', 'linear_out.bias']
        
    def get_data(self):
        return self.dataset

    def get_config(self):
        return self.config

    def store_parameters(self):
        self.keep_weight = deepcopy(self.model.state_dict())
        self.weight_name = list(self.keep_weight.keys())
        self.weight_len = len(self.keep_weight)
        self.fast_weights = OrderedDict()

    def forward(self, support_set_x, support_set_y, query_set_x, num_local_update):
        
        for idx in range(num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            # get all the current model weights and biases
            weight_for_local_update = list(self.model.state_dict().values())
            # compute loss
            support_set_y_pred = self.model(support_set_x)
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            # set all the gradients to zero
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            # local update
            for i in range(self.weight_len):
                if self.weight_name[i] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                else:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
        
        # get final prediction on query_x
        self.model.load_state_dict(self.fast_weights)
        query_set_y_pred = self.model(query_set_x)
        self.model.load_state_dict(self.keep_weight)
        return query_set_y_pred

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, num_local_update):
        batch_sz = len(support_set_xs)
        losses_q = []
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        # computes lose in every set
        for i in range(batch_sz):
            query_set_y_pred = self.forward(support_set_xs[i], support_set_ys[i], query_set_xs[i], num_local_update)
            loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
            losses_q.append(loss_q)
        # concatenates all loses in the set
        losses_q = torch.stack(losses_q).mean(0)
        tqdm.write(str(losses_q))
        self.meta_optim.zero_grad()
        losses_q.backward()
        self.meta_optim.step()
        self.store_parameters()
    
    
    def set_user(self, user_info, items_x, items_y, num_local_update=500):
        self.user_info = torch.tensor(user_info).long().unsqueeze(0)
        
        # create query x
        query_x = None
        for _, item in items_x.iterrows():
            item_info = torch.tensor(item.array).long().unsqueeze(0)
            # concatenates item and user info and add to query x
            items_x_converted = torch.cat((item_info, self.user_info), 1)
            try:
                query_x = torch.cat((query_x, items_x_converted), 0)
            except:
                query_x = items_x_converted
        
        # create support x
        support_x = None
        # get features of every item rated by the user
        for _, item in items_x.iterrows():
            item_info = item.array
            item_info = torch.tensor(item_info).long().unsqueeze(0)
            # concatenates item and user info and add to support x
            items_x_converted = torch.cat((item_info, self.user_info), 1)
            try:
                support_x = torch.cat((support_x, items_x_converted), 0)
            except:
                support_x = items_x_converted
        
        # convert support_y to torch tensor
        support_y = torch.FloatTensor(items_y)
        
        # get prediction for every item, convert to list and limit floats to two decimal points
        query_y_pred = self.forward(support_x, support_y, query_x, num_local_update)
        query_y_pred = query_y_pred.view(-1).tolist()
        
        return np.array(query_y_pred)
        

    def predict(self, items_to_predict):
        items_to_predict = items_to_predict.drop(columns=['age','gender','occupation','user_id'])
        
        # create query x
        query_x = None
        for _, item in items_to_predict.iterrows():
            item_info = torch.tensor(item.array).long().unsqueeze(0)
            # concatenates item and user info and add to query x
            items_x_converted = torch.cat((item_info, self.user_info), 1)
            try:
                query_x = torch.cat((query_x, items_x_converted), 0)
            except:
                query_x = items_x_converted
        
        # get prediction for every item
        self.model.load_state_dict(self.fast_weights)
        query_y_pred = self.model(query_x)
        # convert to list and limit floats to two decimal points
        query_y_pred = query_y_pred.view(-1).tolist()
        # query_y_pred = [round(rate,4) for rate in query_y_pred]
        return np.array(query_y_pred)
