from fastai import *
from fastai.vision import *

## sort of regularization

bs = 64
path = untar_data(URLs.PETS) / "images"
tfms = get_transforms(
    max_rotate=20,
    max_zoom=1.3,
    max_lightining=0.4,
    max_warp=0.4,
    p_affine=1.0,
    p_lighting=1.0,
)

## reflection is usually better

src = ImageItemList.from_folder(path).random_split_by_pct(0.2, seed=42)


def get_data(size, bs, padding_mode="reflection"):
    return (
        src.label_from_re(r"([^/]+)_\d+.jpg$")
        .transform(tfms, size=size, padding_mode=padding_mode)
        .data_bunch(bs=bs)
        .normalize(imagenet_stats)
    )


## fastai function plot_multi()
learn = create_cnn(data, models.resnet34, metrics=erro_rate, bn_final=True)

## heatmap show_heatmaps

