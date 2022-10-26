import torch
from torch.autograd import Variable
from pandas.api.types import is_categorical_dtype

class item(torch.nn.Module):
    def __init__(self, config, items_data):
        super(item, self).__init__()

        # get max value from every feature in item data
        self.features_max_value = []
        for col in items_data.columns:
            if is_categorical_dtype(items_data[col]):
                self.features_max_value.append(items_data[col].nunique())
            else:
                self.features_max_value.append(items_data[col].max()+1)
        # output dimension from every feature
        self.item_num_cols = items_data.shape[1]
        self.embedding_dim = config['embedding_dim']

        # set embedding for every feature
        self.embedding_feature = []
        for i in range(self.item_num_cols):
            self.embedding_feature.append(
                torch.nn.Embedding(
                    num_embeddings=self.features_max_value[i], 
                    embedding_dim=self.embedding_dim
                )
            )

    def forward(self, items):
        # create torch variable for every feature
        items_col = []
        for i in range(self.item_num_cols):
            items_col.append(Variable(items[:, i], requires_grad=False))
        # get embedding of every feature and concat with previous embedding
        items_emb = None
        for i, value in enumerate(items_col):
            try:
                emb = self.embedding_feature[i](value)
            except:
                print(items[0])
                raise ValueError
            try:
                items_emb = torch.cat((items_emb, emb), 1)
            except:
                items_emb = emb
        return items_emb


class user(torch.nn.Module):
    def __init__(self, config, users_data):
        super(user, self).__init__()

        # get max value from every feature in user data
        self.features_max_value = []
        for col in users_data.columns:
            if is_categorical_dtype(users_data[col]):
                self.features_max_value.append(users_data[col].nunique())
            else:
                self.features_max_value.append(users_data[col].max()+1)
        # output dimension from every feature
        self.embedding_dim = config['embedding_dim']
        self.user_num_cols = users_data.shape[1]

        # set embedding for every feature
        self.embedding_feature = []
        for i in range(users_data.shape[1]):
            self.embedding_feature.append(
                torch.nn.Embedding(
                    num_embeddings=self.features_max_value[i], 
                    embedding_dim=self.embedding_dim
                )
            )

    def forward(self, users):
        # create torch variable for every feature
        users_col = []
        for i in range(self.user_num_cols):
            users_col.append(Variable(users[:, i], requires_grad=False))
        # get embedding of every feature and concat with previous embedding
        user_emb = None
        for i, value in enumerate(users_col):
            emb = self.embedding_feature[i](value)
            try:
                user_emb = torch.cat((user_emb, emb), 1)
            except:
                user_emb = emb           
        return user_emb
