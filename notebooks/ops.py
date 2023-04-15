import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from unidecode import unidecode
from wise_geometry.ocr.sorting import sort_objects
from wise_python.parallel import pool_imap
from wise_vision2.ocr.utility import inference_target_computation


def load_menu_data(path):
    dataset_df = pd.read_feather(path)
    print(dataset_df.shape)
    return dataset_df


class ImageMock:
    def __init__(self, content):
        self.size = (content.width, content.height)


def prepare_menu_item_data(content):
    item_types = ["text/wine", "text/cocktail", "text/beer", "text/spirit"]

    description, *_ = inference_target_computation(
        content.labels,
        ImageMock(content),
        item_types + ["text/price"],
        ["text/roi/menu/item"],
    )

    sorted_words = [w for w in sort_objects(description.text_words)]

    word_index_mapping = {
        w.description.word_index: i for i, w in enumerate(sorted_words)
    }

    word_index_to_roi = {}
    for roi_index, roi in enumerate(description.rois):
        for word_index in roi.description.word_indices:
            word_index_to_roi[word_index_mapping[word_index]] = roi_index

    words = []
    words_x_pos = []
    menu_items = []
    visited_rois = set()
    for word_index, word in enumerate(sorted_words):
        roi_index = word_index_to_roi.get(word_index)
        words.append(word.description.text)
        words_x_pos.append(word.shape.centroid.x)

        if roi_index is not None and roi_index not in visited_rois:
            roi = description.rois[roi_index]

            menu_items_words_indices = sorted(
                [word_index_mapping[i] for i in roi.description.word_indices]
            )
            menu_item_entities = [
                o
                for o in description.get_roi_word_entities(roi_index)
                if o.description.name in item_types
            ]

            menu_item_entities_map = {
                word_index_mapping[o.description.word_index]: o
                for o in menu_item_entities
            }

            names = list(set([o.description.name for o in menu_item_entities]))

            item_type = "other"
            if len(names) > 0:
                item_type = names[0].split("/")[-1]

            menu_items_words = [
                menu_item_entities_map[i].description.text
                for i in menu_items_words_indices
                if i in menu_item_entities_map
            ]
            # if not menu_items_words:
            #     continue

            prices = [
                o.description.text
                for o in description.get_roi_word_entities(roi_index)
                if o.description.name == "text/price"
            ]

            menu_items.append(
                (
                    min(menu_items_words_indices),
                    max(menu_items_words_indices),
                    menu_items_words,
                    prices,
                    item_type,
                )
            )
            visited_rois.add(roi_index)

    return content["id"], words, words_x_pos, menu_items


def load_and_prepare_menu_data(path, max_items: int = None):
    dataset_df = load_menu_data(path)
    items = [dataset_df.iloc[content] for content in tqdm(range(len(dataset_df)))]

    random.seed(314)
    random.shuffle(items)
    if max_items is None:
        dataset = pool_imap(prepare_menu_item_data, items[:], num_workers=10)
    else:
        dataset = pool_imap(prepare_menu_item_data, items[:max_items], num_workers=10)
    return dataset


def concatenate_dataset(dataset):
    context_full = []
    targets_full = []
    targets_full_min_max = []

    offset = 0

    for image_id, words, words_pos_x, menu_items in dataset:
        offset = len(context_full)

        for min_index, max_index, menu_item, prices, item_type in menu_items:
            targets_full.append((menu_item, prices, item_type))
            targets_full_min_max.append((min_index + offset, max_index + offset + 1))

        context_full.extend(words)

    targets_full_min_max = np.array(targets_full_min_max)

    return context_full, targets_full, targets_full_min_max


def sample_slice(concatenated_dataset, num_words: int = 80):
    context_full, targets_full, targets_full_min_max = concatenated_dataset

    pos = np.random.randint(0, len(context_full) - num_words)
    words = context_full[pos : pos + num_words]
    targets_indices = np.where(
        (targets_full_min_max[:, 0] >= pos)
        & (targets_full_min_max[:, 1] < pos + num_words)
    )[0]

    context_text = unidecode(" ".join(words).replace(" , ", ", "))
    target_text = ""
    for item_index in targets_indices:
        item_words, item_prices, item_type = targets_full[item_index]
        target_text += f"t>{item_type}\n"
        target_text += f"p>{' '.join(item_prices)}\n"
        target_text += f"i>{' '.join(item_words)}\n"

    target_text = unidecode(target_text).replace(" , ", ", ")

    return context_text, target_text


def create_train_and_eval_loaders(
    dataset,
    keras_tokenizer,
    batch_size: int,
    sequence_length: int,
    num_words: int = 200,
):
    EOT_token = list(keras_tokenizer.get_vocabulary()).index("<|endoftext|>")
    AST_token = list(keras_tokenizer.get_vocabulary()).index("?????")

    validation_data, training_data = dataset[:500], dataset[500:]

    def tf_data_generator(ds):
        concatenated_dataset = concatenate_dataset(ds)
        while True:
            yield sample_slice(concatenated_dataset, num_words=num_words)

    tf_train_dataset = tf.data.Dataset.from_generator(
        lambda: tf_data_generator(training_data),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )

    tf_eval_dataset = tf.data.Dataset.from_generator(
        lambda: tf_data_generator(validation_data),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )

    @tf.function
    def pad_or_slice(tokens, sequence_length: int, constant_values: int = EOT_token):
        tokens = tokens[:sequence_length]
        tokens = tf.pad(
            tokens,
            paddings=[[0, sequence_length - tf.shape(tokens)[0]]],
            constant_values=constant_values,
        )
        tokens.set_shape((sequence_length,))
        return tokens

    @tf.function(
        input_signature=(
            tf.TensorSpec((), tf.string),
            tf.TensorSpec((), tf.string),
        )
    )
    def tokenize_sample(context, target):
        tokens = keras_tokenizer([context, target])
        context = pad_or_slice(tokens[0], sequence_length // 2)
        target = pad_or_slice(
            tf.concat([[AST_token], tokens[1]], 0), sequence_length // 2 + 1
        )

        return context, target

    @tf.function
    def prepare_lm_inputs_labels(context, target):
        x = tf.concat([context, target[:-1]], 0)
        y = tf.concat([context, target[1:]], 0)
        mask = tf.cast(tf.abs(x - y) > 0, tf.int32)
        labels = tf.stack([y, mask], 1)
        return {"inputs_ids": x, "labels": labels}, labels

    # @tf.function
    # def prepare_pretrain_lm_inputs(context, target=None):
    #     context = keras_tokenizer(context)
    #     context = pad_or_slice(context, sequence_length + 1)

    #     x = context[:-1]
    #     y = context[1:]
    #     return x, y

    train_dataset = tf_train_dataset.map(
        tokenize_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.map(
        prepare_lm_inputs_labels, num_parallel_calls=tf.data.AUTOTUNE
    )

    train_dataset = train_dataset.shuffle(2048)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    train_dataset

    eval_dataset = tf_eval_dataset.map(
        tokenize_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    eval_dataset = eval_dataset.map(
        prepare_lm_inputs_labels, num_parallel_calls=tf.data.AUTOTUNE
    )

    eval_dataset = eval_dataset.batch(batch_size)
    eval_dataset = eval_dataset.prefetch(tf.data.AUTOTUNE)
    return train_dataset, eval_dataset


def custom_masked_loss_fn(y_true, y_pred):
    y_true_labels = y_true[..., 0]
    y_true_mask = y_true[..., 1]
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    masked_loss = loss_fn(y_true_labels, y_pred) * tf.cast(y_true_mask, y_pred.dtype)
    return tf.reduce_mean(tf.reduce_mean(masked_loss, axis=-1))
