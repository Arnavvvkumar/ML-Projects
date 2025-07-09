import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
import gradio as gr


RETRAIN = True 
quote           = os.path.expanduser("~/Documents/archive/quotes.csv")
data_dir        = "data"
model_path      = "inspire_model.h5"
tokenizer_path  = "tokenizer.pkl"

seq_len     = 40
embed       = 64
rnn_unit    = 256
batch_size  = 256
buffer_size = 10000
Epochs      = 75
TOP_K       = 3


df = pd.read_csv(quote)
themes = ["courage", "perseverance", "gratitude", "leadership", "mindfulness"]
os.makedirs(data_dir, exist_ok=True)
for theme in themes:
    mask   = df["category"].str.contains(theme, case=False, na=False)
    quotes = df.loc[mask, "quote"].dropna().astype(str)
    with open(os.path.join(data_dir, f"{theme}.txt"), "w", encoding="utf-8") as f:
        for q in quotes:
            f.write(q.strip() + " ")


combined_texts = []
theme_list     = []
for fname in sorted(os.listdir(data_dir)):
    if not fname.endswith(".txt"):
        continue
    t = fname[:-4]
    theme_list.append(t)
    txt = open(os.path.join(data_dir, fname), encoding="utf-8").read().replace("", " ")
    combined_texts.append(f"<theme:{t}> {txt}")
full_text = " ".join(combined_texts)

tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts([full_text])
all_ids = tokenizer.texts_to_sequences([full_text])[0]

vocab_size = len(tokenizer.word_index) + 1
print(f"→ vocab size: {vocab_size}")

inputs, targets = [], []
for i in range(seq_len, len(all_ids)):
    inputs.append(all_ids[i-seq_len : i])
    targets.append(all_ids[i])
inputs = np.array(inputs)
targets = np.array(targets)

inputs  = np.clip(inputs,  0, vocab_size-1)
targets = np.clip(targets, 0, vocab_size-1)
print(f"post-clip max input: {inputs.max()}, max target: {targets.max()}")

dataset = (
    tf.data.Dataset
      .from_tensor_slices((inputs, targets))
      .shuffle(buffer_size)
      .batch(batch_size, drop_remainder=True)
      .cache()
      .prefetch(tf.data.AUTOTUNE)
)


if RETRAIN:
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True),
        LSTM(256, return_sequences=True, unroll=True),
        LSTM(256),
        Dense(vocab_size, activation="softmax")
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["sparse_categorical_accuracy"]
    )
    model.fit(dataset, epochs=Epochs)
    model.save(model_path)
    with open(tokenizer_path, "wb") as f:
        pickle.dump((tokenizer, theme_list), f)
else:
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer, theme_list = pickle.load(f)


def generate_quote_with_suggestions(theme, starting_text, num_words, top_k):

    prefix = starting_text.strip()
    if prefix:
        seed = f"<theme:{theme}> {prefix}"
    else:
        seed = f"<theme:{theme}>"
    token_list = tokenizer.texts_to_sequences([seed])[0]
    result = []
    suggestions_history = []

    for _ in range(num_words):
        padded = pad_sequences([token_list],
                       maxlen=seq_len,
                       padding="post",   
                       truncating="post")

        preds = model.predict(padded, verbose=0)[0]
        topk = tf.math.top_k(preds, k=top_k)
        ids   = topk.indices.numpy()
        probs = topk.values.numpy()

        suggestions = [
            (tokenizer.index_word.get(i, "<unk>"), float(p))
            for i, p in zip(ids, probs)
        ]
        suggestions_history.append(suggestions)

        next_id = ids[0]
        result.append(tokenizer.index_word.get(next_id, "<unk>"))
        token_list.append(next_id)
        token_list = token_list[-seq_len:]

    suggestions_dict = {w: p for w, p in suggestions_history[0]}
    return " ".join(result), suggestions_dict

iface = gr.Interface(
    fn=generate_quote_with_suggestions,
    inputs=[
        gr.Dropdown(choices=theme_list, label="Theme"),
        gr.Textbox(label="Starting words (optional)", placeholder="e.g. Believe in yourself"),
        gr.Slider(5, 50, step=1, value=20, label="Quote Length"),
        gr.Slider(1, 10, step=1, value=TOP_K, label="Number of Suggestions")
    ],
    outputs=["text", "label"],
    title="InspireMe: Motivational Quote Generator",
    description="Generates a quote and shows the top next-word suggestions."
)

if __name__ == "__main__":
    iface.launch(share=True)
