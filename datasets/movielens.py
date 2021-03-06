import os
import re
from datetime import datetime
import pandas as pd


def load_list(path):
    list_ = []
    with open(path, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_


class movielens(object):
    def __init__(self):
        self.name = 'MovieLens'
        self.preprocessed = False
        self.clean_items = None
        self.clean_users = None
        self.all_data = None

        dataset_path = "Datasets/{}".format(self.name)
        self.genre_list = load_list("{}/m_genre.txt".format(dataset_path))
        self.director_list = load_list("{}/m_director.txt".format(dataset_path))
        self.actor_list = load_list("{}/m_actor.txt".format(dataset_path))

        # read data if already exist
        if all([os.path.exists(f'{dataset_path}/{file}_data.csv') for file in ['items', 'users', 'ratings']]):
            self.items_data = pd.read_csv("{}/items_data.csv".format(dataset_path), index_col='id')
            self.users_data = pd.read_csv("{}/users_data.csv".format(dataset_path), index_col='id')
            self.ratings_data = pd.read_csv("{}/ratings_data.csv".format(dataset_path))

        else:  # process data
            profile_data_path = "{}/users.dat".format(dataset_path)
            score_data_path = "{}/ratings.dat".format(dataset_path)
            item_data_path = "{}/movies_extrainfos.dat".format(dataset_path)

            # create items_data
            self.items_data = pd.read_csv(
                item_data_path,
                names=['id', 'title', 'year', 'age rate', 'release date', 'genres',
                       'director', 'writer', 'actors', 'plot', 'poster'],
                sep="::", engine='python', encoding="utf-8"
            )
            self.items_data = self.items_data.drop(columns=['writer', 'plot', 'poster'])

            for genre in self.genre_list:
                self.items_data[genre] = 0
            for director in self.director_list:
                self.items_data[director] = 0
            for actor in self.actor_list:
                self.items_data[actor] = 0
            for idx, row in self.items_data.iterrows():
                for genre in str(row['genres']).split(", "):
                    self.items_data.loc[idx, genre] = 1
                for director in str(row['director']).split(", "):
                    director = re.sub(r'\([^()]*\)', '', director)
                    self.items_data.loc[idx, director] = 1
                for actor in str(row['actors']).split(", "):
                    self.items_data.loc[idx, actor] = 1
            for genre in self.genre_list:
                self.items_data[genre] = self.items_data[genre].astype("bool")
            for director in self.director_list:
                self.items_data[director] = self.items_data[director].astype("bool")
            for actor in self.actor_list:
                self.items_data[actor] = self.items_data[actor].astype("bool")
            self.items_data.set_index('id', inplace=True)

            # create ratings_data
            self.ratings_data = pd.read_csv(
                score_data_path, names=['user_id', 'item_id', 'rating', 'timestamp'],
                sep="::", engine='python'
            )
            self.ratings_data['date'] = self.ratings_data["timestamp"].map(lambda x: datetime.fromtimestamp(x))
            self.ratings_data = self.ratings_data.drop(["timestamp"], axis=1)
            grouped_users = self.ratings_data.groupby('user_id')
            self.ratings_data = grouped_users.filter(lambda x: 12 < len(x) < 25)

            def find_time_from_release(x):
                rate_time = x['date']
                release_time = self.items_data.loc[x['item_id'], 'release date']
                if isinstance(release_time, str):
                    release_time = datetime.strptime(release_time, '%d %b %Y')
                else:
                    release_time = self.items_data.loc[x['item_id'], 'year']
                    release_time = datetime.strptime(str(release_time), '%Y')
                return (rate_time - release_time).days / 365.0
            self.ratings_data['time from release'] = self.ratings_data.apply(find_time_from_release, axis=1)

            # create users_data
            self.users_data = pd.read_csv(
                profile_data_path, names=['id', 'gender', 'age', 'occupation_code', 'zip'],
                sep="::", engine='python'
            )
            users = self.ratings_data['user_id'].unique().tolist()
            self.users_data = self.users_data.loc[self.users_data['id'].isin(users)]
            self.users_data.set_index('id', inplace=True)
            last_users_rate = grouped_users['date'].max()
            self.users_data['last_rate'] = self.users_data.apply(lambda x: last_users_rate[x.name], axis=1)

            # write to files
            self.items_data.to_csv("{}/items_data.csv".format(dataset_path))
            self.users_data.to_csv("{}/users_data.csv".format(dataset_path))
            self.ratings_data.to_csv("{}/ratings_data.csv".format(dataset_path))

    def preprocess(self):
        dataset_path = "Datasets/{}".format(self.name)
        self.clean_items = self.items_data.copy()
        self.clean_users = self.users_data.copy()

        # clean items data
        rate_list = load_list("{}/m_rate.txt".format(dataset_path))
        self.clean_items['age rate'] = self.items_data['age rate'].apply(lambda x: rate_list.index(str(x)))
        self.clean_items['age rate'] = self.clean_items['age rate'].astype("int64")
        self.clean_items['release date'] = self.clean_items['year'].astype("int64")
        self.clean_items = self.clean_items.drop(columns=['title', 'genres', 'year', 'director', 'actors'])

        # clean users data
        gender_list = load_list("{}/m_gender.txt".format(dataset_path))
        age_list = load_list("{}/m_age.txt".format(dataset_path))
        occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
        zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))
        self.clean_users['gender'] = self.users_data['gender'].apply(lambda x: gender_list.index(str(x)))
        self.clean_users['age'] = self.users_data['age'].apply(lambda x: age_list.index(str(x)))
        self.clean_users['occupation_code'] = self.users_data['occupation_code'].apply(
            lambda x: occupation_list.index(str(x)))
        self.clean_users['zip'] = self.users_data['zip'].apply(lambda x: zipcode_list.index(str(x)[:5]))
        self.clean_users = self.clean_users.astype("category")

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
            items_y = items_rated['rating'].reset_index(drop=True)
            user_items = pd.concat([user_x, items_x, items_y], axis=1)
            self.all_data = pd.concat([self.all_data, user_items])

        self.all_data.reset_index(drop=True, inplace=True)
        return self.all_data

    def get_name(self):
        return self.name

    def get_contrast_exp(self, p, q, features):
        diff = ""

        for feature in features:
            if feature in self.genre_list:
                if p[feature]:
                    diff += f'- The recommended movie belongs to the genre of {feature} ' \
                            f'but the movie offered by the user does not.'
                else:
                    diff += f'- The recommended movie does not belong to the {feature} genre ' \
                            f'but the movie suggested by the user does.'

            elif feature in self.actor_list:
                if p[feature]:
                    diff += f'- Actor {feature} plays in the recommended movie ' \
                            f'but not in the movie offered by the user.'
                else:
                    diff += f'- Actor {feature} does not play in the recommended movie ' \
                            f'but does play in the movie offered by the user.'

            elif feature in self.director_list:
                if p[feature]:
                    diff += f'- The recommended movie is directed by {feature} ' \
                            f'unlike the movie offered by the user.'
                else:
                    diff += f'- The recommended movie is not directed by {feature} ' \
                            f'unlike the movie offered by the user.'

            elif feature == 'age rate':
                if p[feature] == "UNRATED":
                    diff += f'- The {feature} {q[feature]} of the movie offered by the user ' \
                            f'is less suited to user preferences.<br>'
                else:
                    diff += f'- The {feature} {p[feature]} of the recommended movie ' \
                            f'is more suitable for user preferences.<br>'
            else:
                diff += f'- The recommended movie was released in {p["year"]}, ' \
                        f'movies from this period are more suited to user preferences.<br>'

            return diff
