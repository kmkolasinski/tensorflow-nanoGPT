# tensorflow-nanoGPT

Example code on how to finetune GPT-2 model using Tensorflow and Keras-NLP library.

The target of this repository is to train generative model to extract Named Entities (NE)
from the input prompt text. Where each NE is separated by \n token. Example:

```python

prompt = "CRICKET - LEICESTERSHIRE TAKE OVER AT TOP AFTER INNINGS VICTORY ."
prediction = predictor(prompt)
prediction['outputs'].numpy().decode() == "LEICESTERSHIRE//ORG\n"

```

# Installation

* See [requirements.txt](requirements.txt) file
* Tested with Tensorflow 2.11
* Checkout example notebook [gpt_2_finetune_conll2003.ipynb](gpt_2_finetune_conll2003.ipynb)


# Main Features

* fast training using mixed precision
* even faster training with XLA enabled (jit_compile)
* partial model freezing and basic implementation of LoRA
* fast data preparation by using tokenizer from keras-nlp package (fully compatible with tf.data.Dataset)
* export trained model to SavedModel - whole processing is stored inside TF graph (preprocessing, tokenization and prediction)
* example how to serve model using tensorflow serving
