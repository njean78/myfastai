from fastai import *
from fastai.vision import *

import numpy as np

## center of a face
path = Path("data/biwi_head_pose")
cal = np.genfromtxt(path / "01" / "rgb.cal", skip_footer=6)


def img2txt_name(f):
    return f"{str(f)[:-7]}pose.txt"


## ... conversion stuff...


def get_ip(img, pts):
    return ImagePoints(FlowField(img.size, pts), scale=True)


## target is made of 2 points -> regression

data = (
    ImageFileList.from_folder(path)
    .label_from_func(get_ctr)
    .split_by_valid_func(lambda o: o[0].parent.name == "13")
    .datasets(PointDataset)
    .transform(get_transforms(), tfm_y=True, size=(120, 160))
    .databunch()
    .normalize(imagenet_stats)
)
data.show_batch(3, figsize=(9, 6))

learn = create_cnn(data, models.resnet34)
learn.loss_func = MSELossFlat()

learn.lr_find()
learn.recorder.plot()
