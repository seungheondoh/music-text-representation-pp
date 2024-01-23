import os
import re
import ast
import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

flatten_list = lambda lst: [item for sublist in lst for item in sublist]


def normalize_text(text):
    """
    Text normalization for alphabets and numbers
        args: text: list of label
    """
    pattern = r"[^a-zA-Z0-9\s]"
    removed_text = re.sub(pattern, "", text)
    return removed_text

def _generate_label_map(label_list):
    """
    Generate label merge dict
        args:
            label_list: list of label
        return:
            label_map: Dict (original label: merge label)
    """
    with torch.no_grad():
        embeddings = model.encode(label_list)
    sim_matrix = util.cos_sim(embeddings, embeddings)
    label_map, pass_idx = {}, []
    for idx, instance in enumerate(sim_matrix):
        if idx not in pass_idx:
            tag = label_list[idx]
            candidate_idx = [jdx for jdx, i in enumerate(instance > 0.9) if i and (jdx != idx)]
            candidate = [label_list[index] for index in candidate_idx]
            cluster = candidate + [tag]
            pass_idx.extend(candidate_idx)
            if len(cluster) > 1:
                key_idx = sorted({len(i):idx for idx, i in enumerate(cluster)}.items())[0][1]
                key_item = cluster.pop(key_idx)
                for remove_tag in cluster:
                    label_map[remove_tag] = key_item
    return label_map

def _apply_label_map(aspect_list, label_map):
    """
    Delete synonym with label map
        args:
            aspect_list: list of label
            label_map : label merge dictionary
        return:
            norm_lists: list of normalize label
    """
    norm_lists = []
    for label_list in aspect_list:
        norm_list = []
        for label in label_list:
            label = normalize_text(label)
            if label in label_map:
                norm_list.append(label_map[label])
            else:
                norm_list.append(label)
        norm_lists.append(norm_list)
    return norm_lists


def label_thresholding(df_binary, threshold):
    # drop by label
    df_sum = df_binary.sum(axis=0)
    df_sum = df_sum[df_sum >= threshold]
    df_binary = df_binary[df_sum.index]
    # drop by columns
    df_sum = df_binary.sum(axis=1)
    df_sum = df_sum[df_sum > 0] # more than one annotation
    df_binary = df_binary.loc[df_sum.index]
    
    min_label = df_binary.sum(axis=0).min()
    min_track = df_binary.sum(axis=1).min()
    return df_binary, min_label, min_track

def iterable_drop(df_binary, threshold):
    while True:
        df_binary, min_label, min_track = label_thresholding(df_binary, threshold)
        if (min_label >= threshold) and (min_track > 0):
            print("converge iterable process, ", "| min label:",min_label, "| min track:", min_track, "| annotation shape", df_binary.shape)
            break
    return df_binary

def group_to_list(group):
    return list(group)

def _get_caption_ground_turth(annotations, id_col, query_col):
    df_grouped = annotations.groupby(id_col)[query_col].apply(group_to_list)
    mlb = MultiLabelBinarizer()
    binarys = mlb.fit_transform(list(df_grouped))
    df_binary = pd.DataFrame(
        binarys, index=list(df_grouped.index), columns=mlb.classes_
    )
    track_to_query, query_to_track = {}, {}
    for query in df_binary:
        track_list = list(df_binary[df_binary[query] == 1].index)
        query_to_track[query] = list(set(track_list))

    for idx in range(len(df_binary)):
        instance = df_binary.iloc[idx]
        track_to_query[instance.name] = list(instance[instance==1].index)   
         
    average_track = np.mean([len(i) for i in query_to_track.values()])
    print("average tag per track: ", average_track)
    print("track pool", df_binary.shape[0])
    print("query pool", df_binary.shape[1])
    return df_binary, track_to_query, query_to_track


def _get_label_ground_turth(dataset_name, annotations, id_col, query_col, threshold):
    query_list = annotations[query_col]
    if query_col == "aspect_list":
        unique_label = list(set(flatten_list(query_list)))
    else:
        unique_label = list(set(query_list))
        query_list = [[i] for i in query_list] # warpping for multilabel binarizer
    normalize_label = [normalize_text(label) for label in unique_label]
    label_map_path = os.path.join(os.path.dirname(__file__), "assets", f"{dataset_name}_map.json")
    if os.path.isfile(label_map_path):
        # generate label_map and reuse for all dataset
        label_map = json.load(open(label_map_path, 'r'))
    else:
        label_map = _generate_label_map(normalize_label)
        json.dump(label_map, open(label_map_path, 'w'), indent=4)
    query_list = _apply_label_map(query_list, label_map)
    mlb = MultiLabelBinarizer()
    binarys = mlb.fit_transform(query_list)
    df_binary = pd.DataFrame(
        binarys, index=list(annotations[id_col]), columns=mlb.classes_
    )
    df_binary = df_binary[~df_binary.index.duplicated(keep='first')]
    df_binary = iterable_drop(df_binary, threshold)
    track_to_label, label_to_track = {}, {}
    for label in df_binary:
        track_list = list(df_binary[df_binary[label] == 1].index)
        label_to_track[label] = list(set(track_list))

    for idx in range(len(df_binary)):
        instance = df_binary.iloc[idx]
        track_to_label[instance.name] = list(instance[instance == 1].index)
    average_track = np.mean([len(i) for i in label_to_track.values()])
    print("average tag per track: ", average_track)
    print("track pool", df_binary.shape[0])
    print("query pool", df_binary.shape[1])
    return df_binary, track_to_label, label_to_track

def query_processor(dataset, query):
    df_annotation = dataset.annotations
    dataset_name = dataset.dataset_name
    if query == "caption":
        query_col = dataset.caption_col
        threshold = None
    else: #  query == "tag"
        query_col = dataset.tag_col
        threshold = 10
    
    df_annotation = df_annotation[df_annotation[query_col] != ""] # without items
    if query == "caption":
        df_binary, track_to_query, query_to_track = _get_caption_ground_turth(
            annotations = df_annotation, 
            id_col = dataset.id_col, 
            query_col = query_col
        )
    else:
        df_binary, track_to_query, query_to_track = _get_label_ground_turth(
            dataset_name = dataset_name,
            annotations = df_annotation, 
            id_col = dataset.id_col, 
            query_col = query_col,
            threshold = threshold
        )
    return df_binary, track_to_query, query_to_track