from fastai import *
from fastai.tabular import *

import pandas as pd

path = untar(URLs.ADULT_SAMPLE)
df = pd.read_csv(path / "adult.csv")

train_df, valid_df = df[:-2000].copy(), df[-2000:].copy()

dep_var = ">=50k"
cat_names = ["workclass"]
cont_names = ['age']

procs = [FillMissing, Categorify, Normalize] # pipeline
test = TabularList.from_df(
    df.iloc[800:1000].copy(), path=path, cat_names=cat_names, cont_names=cont_names
)
data = (
    TabularList.from_df(
        df.iloc[800:1000].copy(),
        path=path,
        cat_names=cat_names,
        cont_names=cont_names,
        procs=procs,
    )
    .split_by_idx(list(range(800, 1000)))
    .label_from_df(cols=dep_var)
    .add_test(test, label=0)
    .databunch()
)


learn = get_tabular_learner(data, layers = [200,100], metrics = accuracy)
learn.fit(1, 1e-2)


