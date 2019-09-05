from fastai import *

categorify = Categorify(small_cat_vars, small_cont_vars)
categorify(small_train_df)
categorify(small_test_df, test=True)

fill_missing = FillMissing(small_cat_vars, small_cont_vars)
fill_missing(small_train_df)
fill_missing(small_test_df, test=True)

## or even better
procs = [FillMissing, Categorify, Normalize]
cat_vars = [...]
cont_vars = [...]

data = (
    TabularList.from_df(
        df, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs
    )
    .split_by_idx(valid_idx)
    .label_from_df(cols=dep_var, label_cls=FloatList, log=True)
    .databunch()
)
# log stands for logarithm

y_range = torch.tensor([0, max_log_y], device=defaults.device)

learn = tabular_learner(
    data,
    layers=[1000, 500],
    p=[0.001, 0.01],
    emb_drop=0.04,  # dropout for embedding layer
    y_range=y_range,
    metrics=exp_rmspe,
)

learn.model

