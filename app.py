__all__ = [
    "learn",
    "categories",
    "classify_image",
    "image",
    "label",
    "examples",
    "intf",
]
# Cell
from fastai.vision.all import *
import gradio as gr


# Cell
learn = load_learner("cdm_model.pkl")

# Cell
categories = ("cat", "dog", "forest", "monkey")


# learn.predict returns
# 1. class string "cat", "dog", "forest", "monkey"
# 2. index - 0 for "cat", 1 for "dog", 2 for "forest", 3 for "monkey"
# 3. probablity - list of 4 tensors, match with index to get the probablity of the predicted class
def classify_image(img):
    category, index, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))


# Cell
image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ["cat.jpg", "dog.jpg", "forest.jpg", "monkey.jpg"]

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)
