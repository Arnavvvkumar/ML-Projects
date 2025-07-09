
from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import numpy as np
import pandas as pd
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
tf.get_logger().setLevel('ERROR') 
from sklearn.model_selection import train_test_split

data =  pd.read_csv('/Users/arnavkumar/Downloads/archive/tmdb_5000_movies.csv')

data['genres'] = data['genres'].apply(lambda x: [genre['name'] for genre in json.loads(x)])

unique_genre = set()

for List in data['genres']:
    for g in List:
        unique_genre.add(g)

unique_genre = sorted(unique_genre)

for gen in unique_genre:
    col = 'genre_' + gen.replace(' ','_')
    data[col] = data['genres'].apply(lambda lst: int(gen in lst))

data = data.drop(columns=['genres'])


def prod_company(txt):
    try:
        items = json.loads(txt)
        if len(items) != 0:
            return items[0]['name']
        else:
            return 'Unknown'
    except Exception:
        return 'Unknown'        

data['main_company'] = data['production_companies'].apply(prod_company)

top10 = data['main_company'].value_counts().head(10).index.tolist()

def top_categorization(company):
    if (company in top10):
        return company
    else:
        return 'Other'
    

data['main_company'] = data['main_company'].apply(top_categorization)

data = data.drop(columns=['production_companies'])


label = input("Enter the desired genre: ").strip()
label = 'genre_' + label.replace(' ', '_') 

if label not in data.columns:
    print("That genre isn’t in the dataset.")
    print("Here are some valid options:")
    for col in data.columns:
        if col.startswith('genre_'):
            print("  •", col[6:].replace('_', ' '))
    quit()


y = data.pop(label)


quantitive_data = ['runtime','budget','revenue','popularity','vote_average','vote_count']

category = ['original_language','main_company']

data = data[quantitive_data + category]

data[quantitive_data] = data[quantitive_data].fillna(0)

for cat in category:
    data[cat] = data[cat].fillna('Unknown')


train, test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=50, stratify=y)

feature_cols = [tf.feature_column.numeric_column(c) for c in quantitive_data]

for cat in ['original_language', 'main_company']:
    vocab = data[cat].unique()
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(cat, vocab)
    feature_cols.append(tf.feature_column.indicator_column(cat_col))

model = tf.estimator.DNNClassifier(feature_columns = feature_cols, hidden_units = [128,64,32], n_classes = 2)


def input_fn(features, labels, training = True, batch_size = 256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))
    if training:
        dataset = dataset.shuffle(10000).repeat()

    return dataset.batch(batch_size)

model.train(input_fn = lambda: input_fn(train, y_train, training=True), steps = 100000)

results = model.evaluate(input_fn = lambda: input_fn(test, y_test, training= False))

accuracy = results['accuracy'] * 100

print(f"\nTest-set accuracy: {accuracy:.1f}%\n")
