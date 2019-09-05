from fastai import *

il = ImageList.from_folder(
    path, convert="L"
)  # the last is a Pillow parameter .. gray scale
il[0].show()

sd = il.split_by_folder(train="training", valid="testing")
ll = sd.label_from_folder()
x, y = ll.train[0]

tfms = (
    [*rand_pad(padding=3, size=28, mode="zeros")],
    [],
)  ## no transform for the validation set

ll = ll.transform(tfms)

bs = 128

data = ll.databunch(bs=bs).normalize()

# simple CNN
def conv(ni, nf):
    """
    ni channels coming in, nf channels coming out
    """
    return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)


model = nn.Sequential(
    conv(1, 8),  # 1*28*28 -> 8*14*14
    nn.BatchNorm2d(8),
    nn.ReLU(),
    conv(8, 16),  # 7
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 32),  # 4
    nn.BatchNorm2d(32),
    nn.ReLU(),
    conv(32, 16),  # 2
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 10),  # 10*1*1
    nn.BatchNorm2d(10),
    nn.ReLU(),
    Flatten(),  # -> vector , removes 1*1 grid -> 10 vector
)

learn = Learner(data, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
learn.summary()
learn.lr_find(end_lr=100)
learn.fit_one_cycle(3, max_lr=0.1)


def conv2(ni, nf):
    """
    conv(8, 16),  # 7
    nn.BatchNorm2d(16),
    nn.ReLU(),
    """
    return conv_layer(ni, nf, stride=2)


model = nn.Sequential(
    conv2(1, 8), conv2(8, 16), conv2(16, 32), conv2(32, 16), conv2(16, 10), Flatten()
)

## Resnet
class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf, nf)
        self.conv2 = conv_layer(nf, nf)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


help(res_block)

model = nn.Sequential(
    conv2(1, 8),
    res_block(8),
    conv2(8, 16),
    res_block(16),
    conv2(16, 32),
    res_block(32),
    conv2(32, 16),
    res_block(16),
    conv2(16, 10),
    Flatten(),
)

def con_and_res(ni, nf): return nn.Sequential(conv2(ni, nf), res_block(nf))



## UNET for image restoration
learn = unet_learner(data, models.resnet34, ...)

## image restoration
from PIL import Image, ImageDraw, ImageFont


def crappify(fn, i):
    dest = path_lr / fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exists_ok=True)
    img = PIL.Image.open(fn)
    targ_sz = resize_to(img, 96, use_min=True)
    img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert("RGB")
    w, h = img.size
    q = random.randint(10, 70)
    ImageDraw.Draw(img).text(
        (random.randint(0, w // 2), random.randint(0, h // 2)),
        str(q),
        fill=(255, 255, 255),
    )
    img.save(dest, quality=q)

## to run in parallel
il = ImageItemList.from_folder(path_hr)
parallel(crappify, il.items)

bs, size = 32, 128
arch = models.resnet34


## GANs ... mse optimization does not help
def create_gen_learner():
    return unet_learner(
        data_gen,
        arch,
        wd=wd,
        blur=True,
        norm_type=NormType.Weight,
        self_attention=True,
        y_range=y_range,
        loss_func=loss_gen,
    )

learn_gen = create_gen_learner()

learn_gen.fit_one_cycle(2, pct_start = 0.8) -> loss 0.04

learn_gen.unfreeze()

learn_gen.fit_one_cycle(3, slice(1e-6, 1e-3)) -> loss 0.04
name_gen = "image_gen"
path_gen = path / name_gen
path_gen.mkdir(exist_ok=True)


def save_preds(dl):
    i = 0
    names = dl.dataset.items  ## filename
    for b in dl:
        preds = learn_gen.pred_batch(batch=b, reconstruct=True)
        for o in preds:
            o.save(path_gen / names[i].name)
            i += 1

