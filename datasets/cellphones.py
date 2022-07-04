import pandas as pd


class cellphones(object):
    def __init__(self):
        self.name = 'Cellphones'
        self.preprocessed = False
        self.clean_items = None
        self.clean_users = None
        self.all_data = None

        # read data
        self.items_data = pd.read_csv('datasets/{}/items.csv'.format(self.name), index_col='id')
        self.users_data = pd.read_csv('datasets/{}/users.csv'.format(self.name), index_col='id')
        self.ratings_data = pd.read_csv('datasets/{}/ratings.csv'.format(self.name), index_col=False)

        self.items_data['external memory'] = self.items_data['external memory'].apply(lambda x: str(x) + 'GB')
        self.items_data['internal memory'] = self.items_data['internal memory'].apply(lambda x: str(x) + 'GB')
        self.items_data['main camera'] = self.items_data['main camera'].apply(lambda x: str(x) + ' MP')
        self.items_data['selfi camera'] = self.items_data['selfi camera'].apply(lambda x: str(x) + ' MP')
        self.items_data['battery'] = self.items_data['battery'].apply(lambda x: str(x) + ' mAh')
        self.items_data['screen size'] = self.items_data['screen size'].apply(lambda x: str(x) + '"')
        self.items_data['weight'] = self.items_data['weight'].apply(lambda x: str(x) + ' g')
        self.items_data['popularity'] = self.items_data['popularity'].apply(lambda x: str(x) + '%')
        self.items_data['price'] = self.items_data['price'].apply(lambda x: str(x) + '$')

    def preprocess(self):
        self.clean_items = pd.read_csv('datasets/{}/items.csv'.format(self.name), index_col='id')
        self.clean_users = pd.read_csv('datasets/{}/users.csv'.format(self.name), index_col='id')

        # clean items data
        brands = self.clean_items['brand'].unique().tolist()
        self.clean_items['brand'] = self.clean_items['brand'].apply(lambda x: brands.index(x))
        self.clean_items['brand'] = self.clean_items['brand'].astype("category")
        models = self.clean_items['model'].unique().tolist()
        self.clean_items['model'] = self.clean_items['model'].apply(lambda x: models.index(x))
        self.clean_items['model'] = self.clean_items['model'].astype("category")
        os = self.clean_items['operating system'].unique().tolist()
        self.clean_items['operating system'] = self.clean_items['operating system'].apply(lambda x: os.index(x))
        self.clean_items['operating system'] = self.clean_items['operating system'].astype("category")

        self.clean_items['screen size'] = self.clean_items['screen size'].apply(lambda x: int(x * 10))
        self.clean_items['main camera'] = self.clean_items['main camera'].apply(lambda x: int(x * 10))
        self.clean_items['release date'] = pd.to_datetime(self.items_data['release date']).dt.year

        # clean users data
        country = self.clean_users['country'].unique().tolist()
        self.clean_users['country'] = self.clean_users['country'].apply(lambda x: country.index(x))
        self.clean_users['country'] = self.clean_users['country'].astype("category")
        lang = self.clean_users['lang'].unique().tolist()
        self.clean_users['lang'] = self.clean_users['lang'].apply(lambda x: lang.index(x))
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
        for idx, row in self.clean_users.iterrows():
            items_rated = self.ratings_data.loc[self.ratings_data['user_id'] == idx]
            items_id = items_rated['item_id']

            user_x = pd.DataFrame([row] * len(items_id)).reset_index()
            items_x = self.clean_items.loc[items_id].reset_index(drop=True)
            items_y = items_rated[['time from release', 'rating']].reset_index(drop=True)
            user_items = pd.concat([user_x, items_x, items_y], axis=1)
            self.all_data = pd.concat([self.all_data, user_items])

        self.all_data.reset_index(drop=True, inplace=True)
        return self.all_data

    def get_name(self):
        return self.name

    def get_contrast_exp(self, p, q, features):
        diff = ""
        for feature in features:
            diff += f'- The recommended cellphones {feature} is {p[feature]} ' \
                    f'compared to {q[feature]} in the cellphone offered by the user.<br>'

        return diff
