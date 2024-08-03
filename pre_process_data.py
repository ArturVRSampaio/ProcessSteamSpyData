import json
import pandas as pd


data = pd.read_csv("dataset/steam_spy_data.csv")

columns_to_drop = ["id", "appid", "name", "developer", "publisher", "created_at", "userscore", "genre", "tags",
                   "languages", "price", "discount", "ccu", "average_forever", "average_2weeks",
                   "median_2weeks", "score_rank", "median_forever", "negative", "positive"]

data["languages"] = data["languages"].apply(lambda x: [lang
                                            .replace('_', '')
                                            .replace('-', '')
                                            .replace(',', '')
                                            .replace('[b]', '')
                                            .replace('[/b]', '')
                                            .strip()
                                            .lower() for lang in x.split(', ')] if isinstance(x, str) else [])
one_hot_encoded = pd.get_dummies(data["languages"].apply(pd.Series).stack()).groupby(level=0).sum()
one_hot_encoded = one_hot_encoded.add_prefix('lang_')
data = pd.concat([data, one_hot_encoded], axis=1)

data["genre"] = data["genre"].apply(lambda x: [genre
                                    .replace('_', '')
                                    .replace('-', '')
                                    .replace(',', '')
                                    .replace('[b]', '')
                                    .replace('[/b]', '')
                                    .strip()
                                    .lower() for genre in x.split(', ')] if isinstance(x, str) else [])
one_hot_encoded = pd.get_dummies(data["genre"].apply(pd.Series).stack()).groupby(level=0).sum()
one_hot_encoded = one_hot_encoded.add_prefix('genre_')
data = pd.concat([data, one_hot_encoded], axis=1)

json_data = data['tags'].apply(json.loads)
data = pd.concat([data, data['tags'].apply(lambda x: pd.Series(json.loads(x.replace('_', '')
                                                                          .replace('-', '')
                                                                          .replace('[b]', '')
                                                                          .replace('[/b]', '')
                                                                          .strip()
                                                                          .lower()))).add_prefix('tag_')], axis=1)
tag_columns = data.filter(like='tag_')
normalized_tag_columns = tag_columns.div(tag_columns.sum(axis=1), axis=0) * 100
normalized_tag_columns = normalized_tag_columns.round(2)
data.update(normalized_tag_columns)

data['user_score'] = ((data['positive']) / (data['positive'] + data['negative'])) * 100

data = data.drop(columns=columns_to_drop)
data = data.fillna(0)


data.to_csv('dataset/output.csv', index=False)
