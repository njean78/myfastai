from fastai import *
from fastai.vision import *

import pandas as pd


# multilabel classification
# data from kaggle

path = Config.data_path() / "planet"
path.mkdir(exists_ok=True)


## filename, label
df = pd.read_csv(path / "train_v2.csv")
df.head()

tfms = get_transforms(flip_vert=true, max_lighting=0.1, max_zoom=1.05, max_warp=0.0)
np.random.seed(42)
## read images and tags
src = (
    ImageFileList.from_folder(path)
    .label_from_csv("train_v2.csv", sep=" ", folder="train_jpg", suffix=".jpg")
    .random_split_by_pct(0.2)
)

## create the dataloader (Data block API)
## data_block.ipynb in doc_src
data = src.datasets().transfrom(tfms, size=128).databunch().normalize(imagenet_stats)

data.show_batch(rows=3, figsize=(10, 9))

data.train_ds[0]
data.valid_ds[0]

## create the learner
arch = model.resnet50
acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)
learn = create_cnn(data, arch, metrics(acc_02, f_score))

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(5, slice(0.01))
learn.save("train-1-rn50")

learn.unfreeze()
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(5, slice(1e-5, 0.01 / 5))
learn.save("train-2-rn50")

### bigger image
data = src.datasets().transfrom(tfms, size=256).databunch().normalize(imagenet_stats)
learn.data = data
data.train_ds[0][0].shape

learn.freeze()
learn.lr_find()
learn.recorder.plot()

lr = 1e-2 / 2

learn.fit_one_cycle(5, slice(lr))
learn.save("train-3-256-rn50")

learn.unfreeze()
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(5, slice(1e-5, lr / 5))
learn.save("train-4-256-rn50")

