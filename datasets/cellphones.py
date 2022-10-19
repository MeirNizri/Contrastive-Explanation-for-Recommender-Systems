import pandas as pd
from datetime import datetime


class cellphones(object):
    def __init__(self):
        self.name = 'Cellphones'
        self.clean_data = None
        self.all_data = None
        self.brands = None
        self.os = None
        self.preprocessed = False

        # read data
        self.cellphones_data = pd.read_csv('datasets/cellphones data.csv', index_col='cellphone_id')

        # add extension to data to make it more readable
        self.cellphones_data['internal memory'] = self.cellphones_data['internal memory'].apply(lambda x: str(x) + 'GB')
        self.cellphones_data['RAM'] = self.cellphones_data['RAM'].apply(lambda x: str(x) + 'GB')
        self.cellphones_data['main camera'] = self.cellphones_data['main camera'].apply(lambda x: str(x) + ' MP')
        self.cellphones_data['selfie camera'] = self.cellphones_data['selfie camera'].apply(lambda x: str(x) + ' MP')
        self.cellphones_data['battery size'] = self.cellphones_data['battery size'].apply(lambda x: str(x) + ' mAh')
        self.cellphones_data['screen size'] = self.cellphones_data['screen size'].apply(lambda x: str(x) + '"')
        self.cellphones_data['weight'] = self.cellphones_data['weight'].apply(lambda x: str(x) + ' g')
        # self.cellphones_data['popularity'] = self.cellphones_data['popularity'].apply(lambda x: str(x) + '%')
        self.cellphones_data['price'] = self.cellphones_data['price'].apply(lambda x: '$' + str(x))

    def preprocess(self):
        if self.preprocessed:
            return self

        self.clean_data = pd.read_csv('datasets/cellphones data.csv')
        self.clean_users = pd.read_csv('datasets/cellphones users.csv')
        self.clean_ratings = pd.read_csv('datasets/cellphones ratings.csv')

        # clean items data
        self.brands = self.clean_data['brand'].unique().tolist()
        self.os = self.clean_data['operating system'].unique().tolist()
        for col in self.brands + self.os:
            self.clean_data[col] = 0
        for idx, row in self.clean_data.iterrows():
            self.clean_data.loc[idx, row['brand']] = 1
            self.clean_data.loc[idx, row['operating system']] = 1
        for col in self.brands + self.os:
            self.clean_data[col] = self.clean_data[col].astype("bool")
        self.clean_data = self.clean_data.drop(columns=['brand', 'model', 'operating system'])

        date_to_int = lambda date: datetime.strptime(date, '%d/%m/%Y').year
        # date_to_int = lambda date: int(datetime.strptime(date, '%b-%y').timestamp())
        self.clean_data['release date'] = self.clean_data['release date'].apply(date_to_int)

        # clean users data
        gender = self.clean_users['gender'].unique().tolist()
        self.clean_users['gender'] = self.clean_users['gender'].apply(lambda x: gender.index(x))
        self.clean_users['gender'] = self.clean_users['gender'].astype("category")
        occupation = self.clean_users['occupation'].unique().tolist()
        self.clean_users['occupation'] = self.clean_users['occupation'].apply(lambda x: occupation.index(x))
        self.clean_users['occupation'] = self.clean_users['occupation'].astype("category")
        
        # self.clean_data.set_index('cellphone_id', inplace=True)
        # self.clean_users.set_index('user_id', inplace=True)
        
        self.preprocessed = True
        return self

    def get_data(self):
        return self.cellphones_data

    def get_clean_data(self):
        if not self.preprocessed:
            self.preprocess()
        return self.clean_data
    
    def get_clean_users(self):
        if not self.preprocessed:
            self.preprocess()
        return self.clean_users
    
    def get_all_data(self):
        if self.all_data is None:
            self.all_data = pd.read_csv('datasets/all data.csv')

        # if not self.preprocessed:
        #     self.preprocess()
            
        # self.all_data = pd.DataFrame()
        # for idx, row in self.clean_users.iterrows():
        #     items_rated = self.clean_ratings.loc[self.clean_ratings['user_id'] == idx]
        #     items_id = items_rated['cellphone_id']

        #     user_x = pd.DataFrame([row] * len(items_id)).reset_index()
        #     items_x = self.clean_data.loc[items_id].reset_index()
        #     items_y = items_rated[['rating']].reset_index(drop=True)
        #     user_items = pd.concat([items_x, user_x, items_y], axis=1)
        #     self.all_data = pd.concat([self.all_data, user_items])

        # self.all_data.reset_index(drop=True, inplace=True)
        
        # self.all_data.to_csv('datasets/all data.csv')
        return self.all_data

    def get_name(self):
        return self.name

    def get_contrast_exp(self, p, q, features, max_len=3):
        brands_picked = False
        os_picked = False
        exp_len = 0
        diff = ""
        for feature in features:

            if feature in self.brands and not brands_picked:
                diff += f'- The recommended cellphone brand is {p["brand"]}, ' \
                        f'compared to {q["brand"]}, which is the brand of the cellphone that Alex preferred.<br>'
                brands_picked = True
                exp_len += 1
                        
            elif feature in self.os and not os_picked:
                diff += f'- The recommended cellphone operating system is {p["operating system"]}, ' \
                        f'compared to {q["operating system"]}, which is the operating system of the cellphone that Alex preferred.<br>'
                os_picked = True
                exp_len += 1
                
            elif feature == 'price':
                diff += f'- The recommended cellphone costs {p[feature]}, ' \
                        f'compared to the cellphone that Alex preferred, which costs {q[feature]}.<br>'
                exp_len += 1

            elif feature in ['RAM', 'internal memory', 'main camera',
                             'selfie camera', 'battery size', 'screen size']:
                diff += f'- The recommended cellphone {feature} is {p[feature]}, ' \
                        f'compared to {q[feature]} in the cellphone that Alex preferred.<br>'
                exp_len += 1

            elif feature == 'weight':
                diff += f'- The recommended cellphone weighs {p[feature]}, ' \
                        f'compared to a weight of {q[feature]} for the cellphone that Alex preferred.<br>'
                exp_len += 1

            elif feature == 'release date':
                diff += f'- The recommended cellphone was released in {p[feature]}, ' \
                        f'while the the cellphone that Alex preferred was released in {q[feature]}.<br>'
                exp_len += 1

            elif feature == 'performance':
                diff += f'- The recommended cellphone performance is rated {p[feature]}, ' \
                        f'compared to {q[feature]} for the cellphone that Alex preferred.<br>'
                exp_len += 1
            
            elif feature == 'model':
                diff += f'- The recommended cellphone {feature} is {p[feature]}, ' \
                        f'compared to {q[feature]} in the cellphone that Alex preferred.<br>'
                exp_len += 1
            
            if exp_len >= max_len:
                break
        
        return diff
