from datetime import datetime
import re
import pandas as pd
  
class movielens(object):
    def __init__(self):
        self.name = 'MovieLens'
        self.preprocessed = False
        self.all_data = None
        # read data
        path = "Datasets/{}".format(self.name)
        profile_data_path = "{}/users.dat".format(path)
        score_data_path = "{}/ratings.dat".format(path)
        item_data_path = "{}/movies_extrainfos.dat".format(path)


        self.items_data = pd.read_csv(
            item_data_path,
            names=['id', 'title', 'year', 'rate', 'released', 'genre', 
                   'director', 'writer', 'actors', 'plot', 'poster'],
            sep="::", engine='python', encoding="utf-8"
        )
        self.items_data.set_index('id', inplace=True)
        

        self.ratings_data = pd.read_csv(
            score_data_path, names=['user_id', 'item_id', 'rating', 'timestamp'],
            sep="::", engine='python'
        )
        self.ratings_data['date'] = self.ratings_data["timestamp"].map(lambda x: datetime.fromtimestamp(x))
        self.ratings_data = self.ratings_data.drop(["timestamp"], axis=1)
        def find_time_from_release(x):
          rate_time = x['date']
          release_time = self.items_data.loc[x['item_id'], 'released']
          if isinstance(release_time, str):
              release_time = datetime.strptime(release_time, '%d %b %Y')
          else:
            release_time = self.items_data.loc[x['item_id'], 'year']
            release_time = datetime.strptime(str(release_time), '%Y')
          return (rate_time - release_time).days/365.0
        self.ratings_data['time_from_release'] = self.ratings_data.apply(find_time_from_release, axis=1)
        grouped_users = self.ratings_data.groupby('user_id')
        self.ratings_data = grouped_users.filter(lambda x: len(x) > 12 and len(x) < 100)


        self.users_data = pd.read_csv(
            profile_data_path, names=['id', 'gender', 'age', 'occupation_code', 'zip'],
            sep="::", engine='python'
        )
        users = self.ratings_data['user_id'].unique().tolist()
        self.users_data = self.users_data.loc[self.users_data['id'].isin(users)]
        self.users_data.set_index('id', inplace=True)
        last_users_rate = grouped_users['date'].max()
        self.users_data['last_rate'] = self.users_data.apply(lambda x: last_users_rate[x.name], axis=1)


    def load_list(self, fname):
        list_ = []
        with open(fname, encoding="utf-8") as f:
            for line in f.readlines():
                list_.append(line.strip())
        return list_

    def preprocess(self):
        self.clean_items = self.items_data.copy()
        self.clean_users = self.users_data.copy()
        dataset_path = "Datasets/{}".format(self.name)

        # clean items data
        rate_list = self.load_list("{}/m_rate.txt".format(dataset_path))
        genre_list = self.load_list("{}/m_genre.txt".format(dataset_path))
        director_list = self.load_list("{}/m_director.txt".format(dataset_path))
        actor_list = self.load_list("{}/m_actor.txt".format(dataset_path))
        self.clean_items['rate'] = self.items_data['rate'].apply(lambda x: rate_list.index(str(x)))
        for genre in genre_list:
            self.clean_items[genre] = 0
        for director in director_list:
            self.clean_items['director_'+str(director)] = 0
        # for actor in actor_list:
        #     self.clean_items['actor_'+str(actor)] = 0
        for idx, row in self.clean_items.iterrows():
            for genre in str(row['genre']).split(", "):
                self.clean_items.loc[idx,genre] = 1
            for director in str(row['director']).split(", "):
                director = re.sub(r'\([^()]*\)', '', director)
                self.clean_items.loc[idx, 'director_'+director] = 1
            # for actor in str(row['actors']).split(", "):
            #     self.clean_items.loc[idx, 'actor_'+actor] = 1
        # self.clean_items['year'] = self.clean_items['year'].apply(lambda x: datetime.strptime(str(x), '%Y'))
        self.clean_items = self.clean_items.drop(columns=['title','released','genre','director','writer','actors','plot','poster'])
        self.clean_items = self.clean_items.astype("category")

        # clean users data
        gender_list = self.load_list("{}/m_gender.txt".format(dataset_path))
        age_list = self.load_list("{}/m_age.txt".format(dataset_path))
        occupation_list = self.load_list("{}/m_occupation.txt".format(dataset_path))
        zipcode_list = self.load_list("{}/m_zipcode.txt".format(dataset_path))
        self.clean_users['gender'] = self.users_data['gender'].apply(lambda x: gender_list.index(str(x)))
        self.clean_users['age'] = self.users_data['age'].apply(lambda x: age_list.index(str(x)))
        self.clean_users['occupation_code'] = self.users_data['occupation_code'].apply(lambda x: occupation_list.index(str(x)))
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
        items_rated = self.ratings_data.loc[self.ratings_data['user_id']==idx]
        items_id = items_rated['item_id']
        
        user_x = pd.DataFrame([row]*len(items_id)).reset_index()
        items_x = self.clean_items.loc[items_id].reset_index(drop=True)
        items_y = items_rated['rating'].reset_index(drop=True)
        user_items = pd.concat([user_x, items_x, items_y], axis=1)
        self.all_data = pd.concat([self.all_data, user_items])

      self.all_data.reset_index(drop=True, inplace=True)
      return self.all_data

    def get_name(self):
        return self.name