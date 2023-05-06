# tensorflow-nanoGPT

Example code on how to finetune GPT-2 model using Tensorflow and Keras-NLP library then
export it to a fully end-to-end model i.e. text-in => text-out in a SavedModel format, which
later can be served with tensorflow serving. Whole processing is stored in the tensorflow graph,
so no extra libraries / tokenizers are needed to run the inference.

The target of this project was to train a generative model to extract Named Entities (NE)
from the input prompt text using model trained on [conll2003](https://huggingface.co/datasets/conll2003) dataset.
The output of this model can be later post processed for further logic.

At the end of the notebook you can run trained model in the following way:

```python
import tensorflow as tf
predictor = tf.saved_model.load('/path/to/gpt2/model')

prompt = "CRICKET - LEICESTERSHIRE TAKE OVER AT TOP AFTER INNINGS VICTORY ."
prediction = predictor(prompt)
prediction['outputs'].numpy().decode() == "LEICESTERSHIRE//ORG\n"

```

# Main Features

* fast training using **mixed precision**
* even faster training with **XLA enabled (jit_compile)**
* partial model freezing and basic implementation of **LoRA**
* **fast data preparation** by using tokenizer from keras-nlp package (fully compatible with tf.data.Dataset)
* **faster token generation with cached keys/values** tensors of attention head
* export trained model to SavedModel - whole processing is stored inside TF graph (preprocessing, tokenization and prediction)
* example how to serve model using **tensorflow serving**


# Some numbers on Google Colab

* Tested on Tesla T4
* I used single but the same prompt for each configuration of the exported model
* GPT-2 - with 256 sequence length

    | Run Type                | Generation time  |
    |-------------------------|------------------|
    | Baseline                | 579 ms ± 38.5 ms |
    | Baseline + XLA          | 369 ms ± 4.37 ms |
    | Cached Key/Values       | 688 ms ± 125 ms  |
    | Cached Key/Values + XLA | 245 ms ± 7.72 ms |


# Installation

* See [requirements.txt](requirements.txt) file
* Tested with Tensorflow 2.11
* Checkout example notebook [gpt_2_finetune_conll2003.ipynb](gpt_2_finetune_conll2003.ipynb)

# Disclaimer

* the aim of this project was not to create any form of SOTA model,
* this is just a test / demo of various features of TensorFlow library,
* the notebook shows how to go from data to production ready and servable model,
* I'm aware that there are still many things which to try to improve the throughput and memory usage.
