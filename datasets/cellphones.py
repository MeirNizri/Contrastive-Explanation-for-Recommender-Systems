import pandas as pd


class cellphones(object):
    def __init__(self):
        self.name = 'Cellphones'
        self.preprocessed = False
        self.all_data = None
        # read data
        path = "datasets/{}".format(self.name)
        self.items_data = pd.read_csv('{}/items.csv'.format(path), index_col='id')
        self.users_data = pd.read_csv('{}/users.csv'.format(path), index_col='id')
        self.ratings_data = pd.read_csv('{}/ratings.csv'.format(path), index_col=False)

        # data_url = 'https://raw.githubusercontent.com/MeirNizri/Contrastive-Explanation-for-Recommender-Systems/main/Datasets/Cellphones/' self.items_data = pd.read_csv(data_url+'items.csv', index_col='id')
        # self.users_data = pd.read_csv(data_url+'users.csv', index_col='id')
        # self.ratings_data = pd.read_csv(data_url+'ratings.csv', index_col=False)

    def preprocess(self):
        self.clean_items = self.items_data.copy()
        self.clean_users = self.users_data.copy()

        # clean items data
        brands = self.items_data['Brand'].unique().tolist()
        self.clean_items['Brand'] = self.items_data['Brand'].apply(lambda x: brands.index(x))
        self.clean_items['Brand'] = self.clean_items['Brand'].astype("category")
        models = self.items_data['Model'].unique().tolist()
        self.clean_items['Model'] = self.items_data['Model'].apply(lambda x: models.index(x))
        self.clean_items['Model'] = self.clean_items['Model'].astype("category")
        os = self.items_data['Operating system'].unique().tolist()
        self.clean_items['Operating system'] = self.items_data['Operating system'].apply(lambda x: os.index(x))
        self.clean_items['Operating system'] = self.clean_items['Operating system'].astype("category")

        self.clean_items['Screen size(inc)'] = self.items_data['Screen size(inc)'].apply(lambda x: int(x * 10))
        self.clean_items['Main Camera(MP)'] = self.items_data['Main Camera(MP)'].apply(lambda x: int(x * 10))
        self.clean_items['Release date'] = pd.to_datetime(self.items_data['Release date']).dt.year

        # clean users data
        country = self.users_data['country'].unique().tolist()
        self.clean_users['country'] = self.users_data['country'].apply(lambda x: country.index(x))
        self.clean_users['country'] = self.clean_users['country'].astype("category")
        lang = self.users_data['lang'].unique().tolist()
        self.clean_users['lang'] = self.users_data['lang'].apply(lambda x: lang.index(x))
        self.clean_users['lang'] = self.clean_users['lang'].astype("category")

        self.preprocessed = True
        return self

    def get_data(self):
        return self.items_data, self.users_data, self.ratings_data

    def get_clean_data(self):
        if not self.preprocessed:
            self.preprocess()
        return self.clean_items, self.clean_users, self.ratings_data

    def get_all_data(self):
        if self.all_data is not None:
            return self.all_data

        self.all_data = pd.DataFrame()
        for idx, row in self.users_data.iterrows():
            items_rated = self.ratings_data.loc[self.ratings_data['user_id'] == idx]
            items_id = items_rated['item_id']

            user_x = pd.DataFrame([row] * len(items_id)).reset_index()
            items_x = self.items_data.loc[items_id].reset_index(drop=True)
            items_y = items_rated['rating'].reset_index(drop=True)
            user_items = pd.concat([user_x, items_x, items_y], axis=1)
            self.all_data = pd.concat([self.all_data, user_items])

        self.all_data.reset_index(drop=True, inplace=True)
        return self.all_data

    def get_name(self):
        return self.name
