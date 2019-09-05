# libffm
# cold start problem ... metadata driven problem
import pandas as pd


path = untar_data(URLs.ML_SAMPLE)
path
ratings = pd.read_csv(path/'ratings.csv')
series2cat(ratings, 'userId', 'movieId')
ratings.head()


learn = get_collab_learner(ratings, n_factors, min_score = 0.0, max_score = 5.0)
learn.fit_one_cycle(4, 5e-3)

data = CollabDataBunch.from_df(rating_movie, seed = 42, pct_val = 0.1, item_name = 'title')
learn = collab_learner(data, n_factors = 40, yrange = [0,5.5], wd = 1e-1)
# wd is usually 0.1...by default it should be 0.01
learn.lr_find()
learn.recorder.plot(skip_end = 15)
learn.fit_one_cycle(5, 5e-3)
# learn.bias user items
# learn.weights