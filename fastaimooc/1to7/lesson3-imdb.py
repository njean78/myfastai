from fastai import *
from fastai.text import *

path = untar_data(URLs.IMDB_SAMPLE)

# tokenize
data_lm = TextDataBunch.from_csv(path, "texts.csv")
data_lm.save()


data = TextDataBunch.load(path)

data.vocab.itos[:10]

## data block API ... alternative approach
data = (
    TextSplitData.from_csv(path, "texts.csv", input_cols=1, label_cols=0, valid_col=2)
    .datasets(TextDataset)
    .tokenize()
    .numericalize()
    .databunch(TextDataBunch)
)

