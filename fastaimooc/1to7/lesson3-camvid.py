from fastai import *
from fastai.vision import *

path = untar_data(URLs.CAMVID)
path.ls()

path_lbl = path/ 'labels'
path_img = path/ 'images'

img_f = fnames[0]
img = open_image(img_f
img.show(figsize = (5,5))

get_y_fn = lambda x : path_lbl/f'{x.stem}_P{x.suffix}''

mask = open_mask(get_y_fn(img_f))
mask.show(figsize = (5,5), alpha = 1)

## CLASSES
codes = np.loadtxt(path/'codes.txt', dtype = str)

## create databunch
src_size = np.array(mask.shape[1:]) ## no channels

size = src_size //2 
bs = 8

src = ImageFileList.from_folder(path_img).label_from_func(get_y_fn).split_by_fname_file('../valid.txt')
# tfm_y -> transform the mask as well
data = src.datasets(SegmentationDataset,classes = codes).transform(get_transforms(), size = size, tfm_y = True).databunch(bs = bs).normalize(imagenet_stats)

data.show_batch(2, figsize = (10,7)) # shows image and mask

name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']
def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()

metrics = acc_camvid
## unet -> segmentation model
learn = Learner.create_unet(data, models.resnet34, metrics= metrics)
lr_find(learn)
learn.recorder.plot()

lr = 1e-2
learn.fit_one_cycle(10, slice(lr))

learn.save('train-1')
learn.load('train-1')

learn.unfreeze()
lr_find(learn)
learn.recorder.plot()

lrs = slice(1e-5, lr/5)

learn.fit_one_cycle(12, lrs)
learn.save('train-2')
learn.recorder.plot_losses()

size = src_size
bs =4

data = src.datasets(SegmentationDataset,classes = codes).transform(get_transforms(), size = size, tfm_y = True).databunch(bs = bs).normalize(imagenet_stats)

## fp16 mixed precision
learn = Learner.create_unet(data, models.resnet34, metrics= metrics).to_fp16()

learn.load('train-2')

lr_find(learn)
learn.recorder.plot()

lr = 1e-3
learn.fit_one_cycle(10, slice(lr))

learn.save('train-3')
learn.load('train-3')

learn.unfreeze()
lr_find(learn)
learn.recorder.plot()

lrs = slice(1e-6, l)

learn.fit_one_cycle(10, lrs)
learn.save('train-4')
learn.recorder.plot_losses()

learn.show_results()