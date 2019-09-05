import numpy as np
from fastai import *
from fastai.vision import *
from fastai.widgets import *


## image download from google images
## go to google images -> ctrl + shifht + j
## paste urls = Array.from(document.querySelectorAll('_rg.di .rg_meta')).map(el => JSON.parse(el.textContent).ou);
##       window.open('data:text/csv;charset=utf-8,' + escape(ursl.join('\n')));
## name it urls_black.txt -> it downloads the urls

classes = ["teddys", "grizzly", "black"]

for clas in classes:
    path = Path("data/bears")
    dest = path / clas
    dest.mkdir(parent=True, exists_ok=True)
    download_images(path / "urls_%s.txt" % clas, dest, max_pics=200)

## cleanup
for clas in classes:
    verify_images(path / clas, delete=True, max_eorkers=8)

np.random.seed(42)
data = ImageDataBunch.from_folder(
    path, train=".", valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=4
).normalize(imagenet_stats)

## visualize
data.classes
data.show_batch(rows=3, figsize=(7, 8))


## Train 1
learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save("train-1")

## Train 2
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(2, max_lr=slice(3e-5, 3e-4))
learn.save("train-1")

## Interpretation
learn.load("train-2")
interp = ClassificationInterpreter.from_learner(learn)
interp.plot_confusion_matrix()

## remove noisy data
losses, idxs = interp.top_losses()
top_loss_paths = data.valid_ds.x[idxs]

fd = FileDeleter(file_paths=top_loss_paths)

## look at 1 singl image
img = open_image(path / "black" / "00000021.jpg")
img

## inference time
fastai.defaults.device = torch.device("cpu")
# path is the path to the model
data2 = ImageDataBunch.single_from_classes(
    path, classes, tfms=get_transforms(), size=224
).normalize(imagenet_stats)
learn = create_cnn(data2, model.resnet34)
learn.load("train-2")


pred_class, pred_idx, output = learn.predict(img)
pred_class

"""
Starlette web app 

@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    img = open_image(BytesIO(bytes))
    _, _, losses = learn.predict(img)
    return JSONResponse(
        {
            "predictions": sorted(
                zip(cat_learner.data.classes, map(float, losses)),
                key=lambda p: p[1],
                reverse=True,
            )
        }
    )
"""

## things can go wrong
# - lr high -> validation loss explosion
# - lr low -> error rate goes down slowly -> learn.recorder.plot_losses() + train loss > val loss
# - few epochs -> train loss >> val loss
# - many epochs -> overfitting

