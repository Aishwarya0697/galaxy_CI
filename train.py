import os
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import tensorflow_hub as hub
import tensorflow_text
import awswrangler as wr
import pandas as pd
import boto3
from multibase import encode, decode
from datetime import datetime, timezone,timedelta
import xtarfile as tarfile


s3 = boto3.client('s3')
obj1 = s3.get_object(Bucket = "galaxy-prod-contentbucket-17ugakz3mt4lq",Key = 'ml-output/users.csv')
obj2 = s3.get_object(Bucket = "galaxy-prod-contentbucket-17ugakz3mt4lq",Key = 'ml-output/interactions.csv')
obj3 = s3.get_object(Bucket = "galaxy-prod-contentbucket-17ugakz3mt4lq",Key = 'ml-output/products.csv')

user = pd.read_csv(obj1['Body'])
interactions = pd.read_csv(obj2['Body'])
product = pd.read_csv(obj3['Body'])

s3_resource = boto3.resource('s3')
my_bucket = s3_resource.Bucket("galaxy-prod-contentbucket-17ugakz3mt4lq")



class UserModel(tf.keras.Model):
  
    def __init__(self, use_timestamps, use_style_tags):
        super().__init__()
        max_tokens = 10000

        self._use_timestamps = use_timestamps
        self._use_style_tags = use_style_tags

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),
        ])

        if use_timestamps:
            self.timestamp_embedding = tf.keras.Sequential([
              tf.keras.layers.Discretization(timestamp_buckets.tolist()),
              tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32),
          ])
            self.normalized_timestamp = tf.keras.layers.Normalization(
              axis=None
          )

            self.normalized_timestamp.adapt(timestamps)

        if use_style_tags:
            self.style_tag_vectorizer = tf.keras.layers.TextVectorization(
              max_tokens=100)

            self.style_tag_embedding = tf.keras.Sequential([
                self.style_tag_vectorizer,
                tf.keras.layers.Embedding(max_tokens, 8), #, mask_zero=True), #This breaks things
                tf.keras.layers.GlobalAveragePooling1D(),
          ])

            self.style_tag_vectorizer.adapt(interactions.map(lambda x: x["user_style_tags"]))

    def call(self, inputs):
        if (self._use_timestamps and self._use_style_tags):
            return tf.concat([
              self.user_embedding(inputs["user_id"]),
              self.timestamp_embedding(inputs["timestamp"]),
              tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1)),
              self.style_tag_embedding(inputs["user_style_tags"]),
          ], axis=1)
          #return self.user_embedding(inputs["user_id"])
        elif self._use_timestamps:
            return tf.concat([
              self.user_embedding(inputs["user_id"]),
              self.timestamp_embedding(inputs["timestamp"]),
              tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1)),
          ], axis=1)
        elif self._use_style_tags:
            return tf.concat([
              self.user_embedding(inputs["user_id"]),
              self.style_tag_embedding(inputs["user_style_tags"]),
          ], axis=1)
        else:
            return tf.concat([
            self.user_embedding(inputs["user_id"]),
          ], axis=1)
      
#productmodel   
class ProductModel(tf.keras.Model):
  
    def __init__(self, use_style_tags_and_description, use_other):
        super().__init__()

        self._use_style_tags_and_description = use_style_tags_and_description
        self._use_other = use_other

        max_tokens = 10000

        self.id_embedding = tf.keras.Sequential([
          tf.keras.layers.StringLookup(
              vocabulary=unique_item_ids, mask_token=None),
          tf.keras.layers.Embedding(len(unique_item_ids) + 1, 32) #, input_length=184)
        ])

        if use_style_tags_and_description:
          #module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
          #module_url = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2"
            module_url = "https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2"
            #def list_head(layer):
            #      def new_layer(ins):
            #          return layer(ins)[0]
            #      return new_layer


            #preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
            #encoder = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1")

            #def default_out(layer, pre):
            #      def new_layer(ins):
            #          return layer(pre(ins))["default"]
            #      return new_layer
            #embed = default_out(encoder, preprocessor) 

            embed = hub.load(module_url)
            #embed = SentenceTransformer('all-MiniLM-L6-v2')


            #embed = hub.KerasLayer(module_url)
            #embed = hub.load("https://tfhub.dev/google/sentence-t5/st5-base/1")

            #self.description_vectorizer = tf.keras.layers.TextVectorization(
            #    max_tokens=max_tokens)

            #self.description_embedding = tf.keras.Sequential([
            #  self.description_vectorizer,
            #  tf.keras.layers.Embedding(max_tokens, 32), #, mask_zero=True), #This breaks things
            #  tf.keras.layers.GlobalAveragePooling1D(),
            #])
            self.description_embedding = embed

            #self.description_vectorizer.adapt(products.map(lambda x: x["description"]))

            self.style_tag_vectorizer = tf.keras.layers.TextVectorization(
                  max_tokens=100)

            self.style_tag_embedding = tf.keras.Sequential([
                self.style_tag_vectorizer,
                tf.keras.layers.Embedding(max_tokens, 8), #, mask_zero=True), #This breaks things
                tf.keras.layers.GlobalAveragePooling1D(),
          ])

            self.style_tag_vectorizer.adapt(products.map(lambda x: x["item_style_tags"]))

            self.currency_id_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_currency_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_currency_ids) + 1, 32)
            ])

            self.category_l2_id_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_category_l2_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_category_l2_ids) + 1, 32)
            ])

            self.category_l3_id_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_category_l3_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_category_l3_ids) + 1, 32)
            ])

            self.brand_id_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_brand_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_brand_ids) + 1, 32)
            ])

            self.size_id_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                  vocabulary=unique_size_ids, mask_token=None),
              tf.keras.layers.Embedding(len(unique_size_ids) + 1, 32)
              ])

            self.gender_id_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
              vocabulary=unique_gender_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_gender_ids) + 1, 1)
            ])

        if use_other:
            self.color_vectorizer = tf.keras.layers.TextVectorization(
              max_tokens=100)

            self.color_id_embedding = tf.keras.Sequential([
                self.color_vectorizer,
                tf.keras.layers.Embedding(max_tokens, 32), #, mask_zero=True), #This breaks things
                tf.keras.layers.GlobalAveragePooling1D(),
              ])

            self.color_vectorizer.adapt(products.map(lambda x: x["color"]))

            self.title_vectorizer = tf.keras.layers.TextVectorization(
                  max_tokens=max_tokens)

            self.title_id_embedding = tf.keras.Sequential([
                self.title_vectorizer,
                tf.keras.layers.Embedding(max_tokens, 32), #, mask_zero=True), #This breaks things
                tf.keras.layers.GlobalAveragePooling1D(),
              ])

            self.title_vectorizer.adapt(products.map(lambda x: x["title"]))


    def call(self, inputs):
    #text_embed = tf.zeros(tf.shape(self.title_text_embedding(titles)))
        embed = self.id_embedding(inputs["item_id"]) 
    #tf.print(text_embed)
        #tf.print(embed)
        if self._use_style_tags_and_description and self._use_other:
              return tf.concat([
                  embed,
                  self.description_embedding(inputs["description"]),
                  self.style_tag_embedding(inputs["item_style_tags"]),
                  self.color_id_embedding(inputs["color"]),
                  self.title_id_embedding(inputs["title"]),
                  self.gender_id_embedding(inputs["gender"]),
                  self.currency_id_embedding(inputs["currency"]),
                  self.category_l2_id_embedding(inputs["category_l2"]),
                  self.category_l3_id_embedding(inputs["category_l3"]),
                  self.brand_id_embedding(inputs["brand"]),
                  self.size_id_embedding(inputs["size"]),
              ], axis=1)
        elif self._use_style_tags_and_description:
              return tf.concat([
                  embed,
                  self.description_embedding(inputs["description"]),
                  self.style_tag_embedding(inputs["item_style_tags"]),
                  self.currency_id_embedding(inputs["currency"]),
                  self.category_l2_id_embedding(inputs["category_l2"]),
                  self.category_l3_id_embedding(inputs["category_l3"]),
                  self.brand_id_embedding(inputs["brand"]),
                  self.size_id_embedding(inputs["size"]),
                  self.gender_id_embedding(inputs["gender"]),
              ], axis=1)
        elif self._use_other:
              return tf.concat([
                  embed,
                  self.color_id_embedding(inputs["color"]),
                  self.title_id_embedding(inputs["title"]),
              ], axis=1)
        else:
              return tf.concat([
                  embed,
              ], axis=1)

            
#galaxymodel

class GalaxyModel(tfrs.models.Model):

    def __init__(self, use_timestamps, use_style_tags_and_description, use_other, unique_user_ids, unique_item_ids, unique_currency_ids, unique_category_l2_ids, unique_category_l3_ids,unique_brand_ids, unique_size_ids, unique_gender_ids, interactions, products):
        super().__init__()
        dense_size = 8 #was 32
        self.query_model = tf.keras.Sequential([
              UserModel(use_timestamps, use_style_tags_and_description),
              tfrs.layers.dcn.Cross(projection_dim=4,kernel_initializer="glorot_uniform"),
              tf.keras.layers.Dense(dense_size, activation="relu"),
              #tf.keras.layers.Dropout(0.99, seed=42, input_shape=(dense_size,)), #doesn't seem to work
              #tf.keras.layers.Dense(dense_size),
              tf.keras.layers.Dense(dense_size) 
                                                #kernel_regularizer=tf.keras.regularizers.L2(10.0))

            ])
        product_model=ProductModel(use_style_tags_and_description, use_other)
        self.candidate_model = tf.keras.Sequential([
              product_model,
              tfrs.layers.dcn.Cross(projection_dim=4,kernel_initializer="glorot_uniform"),
              tf.keras.layers.Dense(dense_size, activation="relu"),
                #, activity_regularizer=tf.keras.regularizers.L1L2(10.0)), #appears to give perfect results?
              #tf.keras.layers.Dropout(0.99, seed=42, input_shape=(dense_size,)), #doesn't seem to work
              tf.keras.layers.Dense(dense_size)
                                                #kernel_regularizer=tf.keras.regularizers.L2(10.0))

            ])

        self.task = tfrs.tasks.Retrieval(
            metrics= tfrs.metrics.FactorizedTopK(#metrics=[tfr.keras.metrics.NDCGMetric(name="ndcg_metric", topn=10)],
                candidates=products.batch(10000).map(self.candidate_model))#, num_hard_negatives=1000 #, remove_accidental_hits=True

        )

    def compute_loss(self, features, training=False):
        # We only pass the user id and timestamp features into the query model. This
        # is to ensure that the training inputs would have the same keys as the
        # query inputs. Otherwise the discrepancy in input structure would cause an
        # error when loading the query model after saving it.
        query_embeddings = self.query_model({
            "user_id": features["user_id"],
            "timestamp": features["timestamp"],
            "user_style_tags": features["user_style_tags"]
        })
        product_embeddings = self.candidate_model({
            "item_id": features["item_id"],
            "description": features["description"],
            "item_style_tags": features["item_style_tags"],
            "currency": features["currency"],
            "category_l2": features["category_l2"],
            "category_l3": features["category_l3"],
            "brand": features["brand"],
            "size": features["size"],
            "color": features["color"],
            "title": features["title"],
            "gender": features["gender"],
        })

        return self.task(query_embeddings, product_embeddings)

#load_data

def load_data(user, product, interactions):
    galaxy = interactions.sort_values(by:=['timestamp']).query(
    'event_type == "highlights_purchase"' +
    ' or event_type == "product_pageView_purchase"'+
    ' or event_type == "seller_profile_purchase"'
    ' or event_type == "product_pageView"')

    #was both user_id and item_id, testing unique items
    #galaxy = galaxy.drop_duplicates()
    #galaxy = galaxy.drop_duplicates(['user_id','item_id'])
    #galaxy = galaxy.drop_duplicates(['item_id'])
    #galaxy = galaxy.drop_duplicates(['user_id'])
    #print(len(galaxy))

    other_id_name = "title"

    galaxyProducts = product.rename(columns={"ITEM_ID": "item_id",
                                                                                            "STYLE_TAGS": "item_style_tags",
                                                                                            "DESCRIPTION": "description",
                                                                                            "CURRENCY": "currency",
                                                                                            "CATEGORY_L2": "category_l2",
                                                                                            "CATEGORY_L3": "category_l3",
                                                                                            "BRAND": "brand",
                                                                                            "SIZE": "size",
                                                                                            "COLOR": "color",
                                                                                            "TITLE": "title",
                                                                                            "GENDER": "gender",
                                                                                            "QUANTITY": "quantity"

                                                                                            })
    galaxyProducts['item_style_tags'] = galaxyProducts['item_style_tags'].str.replace('|',' ',regex=True)
    galaxyProducts['color'] = galaxyProducts['color'].str.replace('|',' ',regex = True)
    galaxyProducts['quantity'] = galaxyProducts['quantity'].fillna(0.0).astype(int)

    galaxy = galaxy.join(galaxyProducts.set_index('item_id'), on="item_id", how="left")

    global galaxyUsers
    galaxyUsers = user.rename(columns={"STYLE_TAGS": "user_style_tags",
                                                                    "USER_ID": "user_id"})

    galaxyUsers['user_style_tags'] = galaxyUsers['user_style_tags'].str.replace('|',' ',regex = True)

    galaxy = galaxy.join(galaxyUsers.set_index('user_id'), on="user_id", how="left")
    print(galaxyProducts)

    galaxy_features = galaxy.fillna("").copy()[["timestamp", "user_id", "item_id", "description",
                                            "currency", "category_l2", "category_l3", "brand", "size",
                                            "item_style_tags", "user_style_tags", "color", "title", "gender", "event_type"]] #"item_id",

    non_purchase_features = galaxy_features.query('event_type == "product_pageView"')

    galaxy_features = galaxy_features.query('event_type != "product_pageView"')#.drop_duplicates(['item_id'])


    galaxy_features_dict = {name: np.array(value)
                            for name, value in galaxy_features.items()}

    features_ds = tf.data.Dataset.from_tensor_slices(galaxy_features_dict)



    non_purchase_galaxy_features_dict = {name: np.array(value)
                            for name, value in non_purchase_features.items()}


    non_purchase_features_ds = tf.data.Dataset.from_tensor_slices(non_purchase_galaxy_features_dict)


    galaxy_product_features = galaxyProducts.copy()[["item_id", "description", "item_style_tags",
                                                    "currency", "category_l2", "category_l3", "brand",
                                                    "size", "color", "title", "gender", "quantity"]].fillna("")


    print(galaxy_product_features)

    galaxy_product_features_dict = {name: np.array(value)
                            for name, value in galaxy_product_features.items()}

    print(galaxy_product_features_dict)
    products = tf.data.Dataset.from_tensor_slices(galaxy_product_features_dict)

    print(galaxy_features.tail)

    interactions = features_ds.map(lambda x: {
    "description": x["description"],
    "user_id": x["user_id"],
    "item_id": x["item_id"],
    "timestamp": x["timestamp"],
    "user_style_tags": x["user_style_tags"],
    "item_style_tags": x["item_style_tags"],
    "currency": x["currency"],
    "category_l2": x["category_l2"],
    "category_l3": x["category_l3"],
    "brand": x["brand"],
    "size": x["size"],
    "color": x["color"],
    "title": x["title"],
    "gender": x["gender"],
    })

    non_purchase_interactions = non_purchase_features_ds.map(lambda x: {
    "description": x["description"],
    "user_id": x["user_id"],
    "item_id": x["item_id"],
    "timestamp": x["timestamp"],
    "user_style_tags": x["user_style_tags"],
    "item_style_tags": x["item_style_tags"],
    "currency": x["currency"],
    "category_l2": x["category_l2"],
    "category_l3": x["category_l3"],
    "brand": x["brand"],
    "size": x["size"],
    "color": x["color"],
    "title": x["title"],
    "gender": x["gender"],
    })
    print(type(features_ds))
    print(type(products))

    purchase_counts = galaxy_features["item_id"].value_counts().to_frame().rename(columns={"item_id": "purchase_count"})

    print(purchase_counts)
    available_product_features = galaxy_product_features.join(purchase_counts, on="item_id", how="left").fillna(0).query('quantity > purchase_count')

    print(galaxy_product_features)
    products = products.map(lambda x: {"description": x["description"], "item_id": x["item_id"],
                                    "item_style_tags": x["item_style_tags"],
                                    "currency": x["currency"],
                                    "category_l2": x["category_l2"],
                                    "category_l3": x["category_l3"],
                                    "brand": x["brand"],
                                    "size": x["size"],
                                    "color": x["color"],
                                    "title": x["title"],
                                    "gender": x["gender"],
                                    })


    galaxy_available_product_features_dict = {name: np.array(value)
                            for name, value in available_product_features.items()}

    print(galaxy_product_features_dict)
    available_products = tf.data.Dataset.from_tensor_slices(galaxy_available_product_features_dict)

    print(available_product_features)

    avaiable_products = available_products.map(lambda x: {"description": x["description"], "item_id": x["item_id"],
                                    "item_style_tags": x["item_style_tags"],
                                    "currency": x["currency"],
                                    "category_l2": x["category_l2"],
                                    "category_l3": x["category_l3"],
                                    "brand": x["brand"],
                                    "size": x["size"],
                                    "color": x["color"],
                                    "title": x["title"],
                                    "gender": x["gender"],
                                    })

    #global all_products
    #all_products = products

    global avail_products
    avail_products = available_products
    print(products)
    print(interactions)

    global unique_available_item_ids
    unique_available_item_ids = np.unique(np.concatenate(list(available_products.batch(1000).map(lambda x: x["item_id"]))))
    unique_item_ids = np.unique(np.concatenate(list(products.batch(1000).map(lambda x: x["item_id"]))))
    unique_user_ids = np.unique(np.concatenate(list(interactions.batch(1000).map(lambda x: x["user_id"]))))
    unique_currency_ids = np.unique(np.concatenate(list(interactions.batch(1000).map(lambda x: x["currency"]))))
    unique_category_l2_ids = np.unique(np.concatenate(list(interactions.batch(1000).map(lambda x: x["category_l2"]))))
    unique_category_l3_ids = np.unique(np.concatenate(list(interactions.batch(1000).map(lambda x: x["category_l3"]))))
    unique_brand_ids = np.unique(np.concatenate(list(interactions.batch(1000).map(lambda x: x["brand"]))))
    unique_size_ids = np.unique(np.concatenate(list(interactions.batch(1000).map(lambda x: x["size"]))))
    unique_gender_ids = np.unique(np.concatenate(list(interactions.batch(1000).map(lambda x: x["gender"]))))

    tf.random.set_seed(42)
    #.shuffle(100, seed=42, reshuffle_each_iteration=False)

    interaction_count = len(interactions)
    print(interaction_count)
    train_count = int(interaction_count * 0.8)
    test_count = int(interaction_count * 0.2)

    train = interactions.take(train_count)
    test = interactions.skip(train_count).take(test_count)

    #train = train.concatenate(non_purchase_interactions)

    cached_train = train.shuffle(train_count, seed=42, reshuffle_each_iteration=True).batch(int(train_count)).cache()
    cached_test = test.batch(test_count).cache()



    return (products, interactions, unique_user_ids, unique_item_ids, unique_currency_ids, unique_category_l2_ids, unique_category_l3_ids, unique_brand_ids, unique_size_ids, unique_gender_ids, cached_train, cached_test)



if __name__ == "__main__":

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_factorized_top_k/top_10_categorical_accuracy', patience=15,      mode="max",restore_best_weights=True) #doesn't work with mode=max 
    products, interactions, unique_user_ids, unique_item_ids, unique_currency_ids, unique_category_l2_ids, unique_category_l3_ids,  unique_brand_ids, unique_size_ids, unique_gender_ids, cached_train, cached_test = load_data(user,product,interactions)

        #so it's reproduceable
    tf.random.set_seed(
        42
    )

        # global model
    model = GalaxyModel(use_timestamps=False, use_style_tags_and_description=True, use_other=True, unique_user_ids=unique_user_ids, unique_item_ids=unique_item_ids, unique_currency_ids=unique_currency_ids, unique_category_l2_ids=unique_category_l2_ids, unique_category_l3_ids=unique_category_l3_ids, unique_brand_ids=unique_brand_ids, unique_size_ids=unique_size_ids, unique_gender_ids=unique_gender_ids, interactions=interactions, products=products)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
    model.fit(cached_train, epochs=10, validation_data=cached_test, callbacks=[callback])

    # train_accuracy = model.evaluate(
    #    cached_train, return_dict=True)["ndcg_metric"] #["factorized_top_k/top_100_categorical_accuracy"]
    # test_accuracy = model.evaluate(
    #    cached_test, return_dict=True)["ndcg_metric"] #["factorized_top_k/top_100_categorical_accuracy"]

    # print(f"Top-10 NDCG (train): {train_accuracy:.4f}.")
    # print(f"Top-10 NDCG (test): {test_accuracy:.4f}.")

    train_accuracy = model.evaluate(
    cached_train, return_dict=True)["factorized_top_k/top_10_categorical_accuracy"]
    test_accuracy = model.evaluate(
    cached_test, return_dict=True)["factorized_top_k/top_10_categorical_accuracy"]

    print(f"Top-10 accuracy (train): {train_accuracy:.4f}.", flush=True)
    print(f"Top-10 accuracy (test): {test_accuracy:.4f}.", flush=True)

    global index
    # Create a model that takes in raw query features, and
    index = tfrs.layers.factorized_top_k.BruteForce(model.query_model)
    # recommends products out of the entire products dataset.
    index.index_from_dataset(
      tf.data.Dataset.zip((avail_products.map(lambda x: x["item_id"]).batch(100), avail_products.batch(100).map(model.candidate_model)))
        )


query_user_id = '9c0c37a1-d1ac-4864-ab5a-91f6a645fe56'

# Create a model that takes in raw query features, and
# index = tfrs.layers.factorized_top_k.BruteForce(model.query_model)
# # recommends products out of the entire products dataset.
# index.index_from_dataset(
#   tf.data.Dataset.zip((products.map(lambda x: x["item_id"]).batch(100), products.batch(100).map(model.candidate_model)))
# )


query_user_features = galaxyUsers.query('user_id == "' + query_user_id + '"')["user_style_tags"].head().values[0]

# Get recommendations.
_, titles = index({"user_id": tf.constant([query_user_id]), 
                                          "user_style_tags": tf.constant([query_user_features])})
print(f"Recommendations for user " + query_user_id + f": {titles[0, :3]}")

results = tf.keras.backend.get_value(titles).tolist()[0]

results = [str(x, 'utf-8') for x in results]

print(type(tf.keras.backend.get_value(titles).tolist()))
print(results)
print(query_user_features)
print(model.query_model.summary())
print(model.candidate_model.summary())

tf.saved_model.save(index, "/opt/ml/model")


zipped_model_path = os.path.join("/opt/ml/model", "model.tar.gz")

with tarfile.open(zipped_model_path, "w:gz") as tar:
    tar.add('/opt/ml/model')

boto3.Session().resource('s3').Bucket("galaxy-prod-contentbucket-17ugakz3mt4lq").Object(os.path.join('ml-output/tmp/', 'model.tar.gz')).upload_file("/opt/ml/model/model.tar.gz")
print("tesing")
print("training")