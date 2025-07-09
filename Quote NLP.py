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

RETRAIN         = True
CSV_PATH        = os.path.expanduser("~/Documents/archive/quotes.csv")
DATA_DIR        = "data"
MODEL_PATH      = "inspire_model.h5"
TOKENIZER_PATH  = "tokenizer.pkl"

SEQ_LEN     = 40
EMBED_DIM   = 64
RNN_UNITS   = 256
BATCH_SIZE  = 256
BUFFER_SIZE = 10000
EPOCHS      = 30
TOP_K       = 3


df = pd.read_csv(CSV_PATH)
themes = ["courage","perseverance","gratitude","leadership","mindfulness"]
os.makedirs(DATA_DIR, exist_ok=True)
for theme in themes:
    mask   = df["category"].str.contains(theme, case=False, na=False)
    quotes = df.loc[mask, "quote"].dropna().astype(str)
    with open(os.path.join(DATA_DIR, f"{theme}.txt"), "w", encoding="utf-8") as f:
        for q in quotes:
            f.write(q.strip() + " ")


combined_texts = []
theme_list     = []
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".txt"):
        continue
    t = fname[:-4]
    theme_list.append(t)
    txt = (
      open(os.path.join(DATA_DIR, fname), encoding="utf-8")
      .read()
      .replace("\n"," ")
    )
    combined_texts.append(f"<theme:{t}> {txt}")

full_text = " ".join(combined_texts)


tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts([full_text])
all_ids = tokenizer.texts_to_sequences([full_text])[0]

vocab_size = len(tokenizer.word_index) + 1
print("→ vocab size:", vocab_size)


inputs, targets = [], []
for i in range(SEQ_LEN, len(all_ids)):
    inputs.append(all_ids[i-SEQ_LEN : i])
    targets.append(all_ids[i])
inputs = np.array(inputs)
targets = np.array(targets)

# clamp just in case
inputs  = np.clip(inputs,  0, vocab_size-1)
targets = np.clip(targets, 0, vocab_size-1)
print("post-clip max input:", inputs.max(), " max target:", targets.max())


dataset = (
    tf.data.Dataset
      .from_tensor_slices((inputs, targets))
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE, drop_remainder=True)
      .cache()
      .prefetch(tf.data.AUTOTUNE)
)


if RETRAIN:
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBED_DIM),   # <- mask_zero removed
        LSTM(RNN_UNITS, return_sequences=True),
        LSTM(RNN_UNITS),
        Dense(vocab_size, activation="softmax")
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["sparse_categorical_accuracy"]
    )
    model.fit(dataset, epochs=EPOCHS)
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump((tokenizer, theme_list), f)
else:
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer, theme_list = pickle.load(f)


def generate_quote_with_suggestions(theme, starting_text, num_words, top_k):
    prefix = starting_text.strip()
    seed = f"<theme:{theme}> {prefix}" if prefix else f"<theme:{theme}>"
    token_list = tokenizer.texts_to_sequences([seed])[0]
    result, suggestions_history = [], []
    for _ in range(num_words):
        padded = pad_sequences([token_list], maxlen=SEQ_LEN, padding="post")
        preds   = model.predict(padded, verbose=0)[0]
        topk    = tf.math.top_k(preds, k=top_k)
        ids     = topk.indices.numpy()
        probs   = topk.values.numpy()
        # record the top-k suggestions at this step
        suggestions = [(tokenizer.index_word.get(i, "<unk>"), float(p)) 
                       for i,p in zip(ids,probs)]
        suggestions_history.append(suggestions)
        # pick the top1 word
        next_id = ids[0]
        result.append(tokenizer.index_word.get(next_id, "<unk>"))
        token_list.append(next_id)
        token_list = token_list[-SEQ_LEN:]
    # only return suggestions from the first generation step
    return " ".join(result), dict(suggestions_history[0])


iface = gr.Interface(
    fn=generate_quote_with_suggestions,
    inputs=[
      gr.Dropdown(choices=theme_list, label="Theme"),
      gr.Textbox(label="Starting words (optional)",
                 placeholder="e.g. Believe in yourself"),
      gr.Slider(5, 50, step=1, value=20, label="Quote Length"),
      gr.Slider(1, 10, step=1, value=TOP_K, label="Number of Suggestions")
    ],
    outputs=["text","label"],
    title="InspireMe: Motivational Quote Generator",
    description="Generates a quote and top-word suggestions."
)

if __name__=="__main__":
    iface.launch(share=True)
