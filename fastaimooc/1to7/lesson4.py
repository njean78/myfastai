from fastai import *
from fastai.text import *

import pandas as pd

path = untar_data(URLs.IMDB_SAMPLE)
path.ls()
# texts.csv, models, tmp
df = pd.read_csv(path / "texts.csv")

data_lm = TextDataBunch.from_csv(path, "texts.csv")
data_lm.save()  # preprocessing ... and saving in path

data = TextDataBunch.load(path)
data.show_batch()

data.vocab.itos[:10]  # default 60000 words
data.train_ds[0][0]
data.train_ds[0][0].data[:10]


## or ... data block API
data = (
    TextList.from_csv(path, "texts.csv", col="text")
    .split_from_df(cols=2)  # validation flag
    .label_from_df(cols=0)  # labels
    .databunch()
)

data_lm = (
    TextList.from_folder(path)
    .filter_by_folder(include=["train", "test"])
    .random_split_by_pct(0.1)
    .label_for_lm()
    .databunch()
)

data_lm.save("tmp_lm")

learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.3)
learn.lr_find()

learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))

learn.save("lm-head.model")
learn.load("lm-head.model")

learn.unfreeze()
learn.fit_one_cycle(10, 1e-3, moms=(0.8, 0.7))

## legal lm -> auc should be 50%

learn.predict("I liked this movie because ", 100, temperature=1.1, min_p=0.001)

learn.save("fine-tuned.lm")
learn.save_encoder(
    "fine-tuned.encoder"
)  # the part that understand the sentence, without the next word generator
## classifier
data_clas = (
    TextFilesList.from_folder(path, vocab=data_lm.vocab)
    .split_by_folder(valid="test")
    .label_from_folder(classe=["neg", "pos"])
    .databunch(bs=50)
)
data_clas.save("tmp_clas")
# data_clas = TextCkasDataBunch.load(path, 'tmp_clas', bs = 50)
data_clas.show_batch()

learn = text_classifier_learner(data_clas, drop_mult=0.5)
learn.load_encodr("fine_tuned_enc")
learn.freeze()

learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(1,2e-2,moms = (0.8,0.7)) # -> .92

learn.save('first')
learn.unfreeze(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2),moms = (0.8,0.7))

learn.unfreeze(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3),moms = (0.8,0.7))

learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3),moms = (0.8,0.7))