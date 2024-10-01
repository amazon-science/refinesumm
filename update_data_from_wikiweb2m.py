import numpy as np
import glob
import tensorflow.compat.v1 as tf
from collections import defaultdict
from tqdm import tqdm
import json 
import pandas as pd
import evaluate
from tqdm import tqdm

class args:
  split = "val" #other splits: "test", "train"
  data_path = "https://huggingface.co/datasets/vaidehi99/RefineSumm/raw/main/refinesumm_{}.csv".format(split)
  out_path = "data/refinesumm_{}_wikiweb2m.csv".format(split)
  
def get_sec_and_img_urls(args):
  parser = DataParser()
  parser.parse_data()
  data = pd.read_csv(args.data_path)
  txt = []
  img = []
  for i in tqdm(range(len(data))):
    cur = parser.data['test'][data["wikiweb2m_idx"][i]]
    img.append(cur[1]['section_image_url'].values[data["img_url_idx"][i]].numpy().decode())
    txt.append(cur[1]['section_text'].values[data["sec_idx"][i]].numpy().decode())
  data['txt'] = txt
  data['img'] = img
  data.insert(len(data.columns)-1, 'summary', data.pop('summary'))
  data.to_csv(args.out_path, index=False)

class DataParser():
  def __init__(self,
               path: str = "data/",
               filepath: str = 'wikiweb2m-*',
               ):
    self.filepath = filepath
    self.path = path
    self.data = defaultdict(list)

  def parse_data(self):
    context_feature_description = {
        'split': tf.io.FixedLenFeature([], dtype=tf.string),
        'page_title': tf.io.FixedLenFeature([], dtype=tf.string),
        'page_url': tf.io.FixedLenFeature([], dtype=tf.string),
        'clean_page_description': tf.io.FixedLenFeature([], dtype=tf.string),
        'raw_page_description': tf.io.FixedLenFeature([], dtype=tf.string),
        'is_page_description_sample': tf.io.FixedLenFeature([], dtype=tf.int64),
        'page_contains_images': tf.io.FixedLenFeature([], dtype=tf.int64),
        'page_content_sections_without_table_list': tf.io.FixedLenFeature([] , dtype=tf.int64)
    }

    sequence_feature_description = {
        'is_section_summarization_sample': tf.io.VarLenFeature(dtype=tf.int64),
        'section_title': tf.io.VarLenFeature(dtype=tf.string),
        'section_index': tf.io.VarLenFeature(dtype=tf.int64),
        'section_depth': tf.io.VarLenFeature(dtype=tf.int64),
        'section_heading_level': tf.io.VarLenFeature(dtype=tf.int64),
        'section_subsection_index': tf.io.VarLenFeature(dtype=tf.int64),
        'section_parent_index': tf.io.VarLenFeature(dtype=tf.int64),
        'section_text': tf.io.VarLenFeature(dtype=tf.string),
        'section_clean_1st_sentence': tf.io.VarLenFeature(dtype=tf.string),
        'section_raw_1st_sentence': tf.io.VarLenFeature(dtype=tf.string),
        'section_rest_sentence': tf.io.VarLenFeature(dtype=tf.string),
        'is_image_caption_sample': tf.io.VarLenFeature(dtype=tf.int64),
        'section_image_url': tf.io.VarLenFeature(dtype=tf.string),
        'section_image_mime_type': tf.io.VarLenFeature(dtype=tf.string),
        'section_image_width': tf.io.VarLenFeature(dtype=tf.int64),
        'section_image_height': tf.io.VarLenFeature(dtype=tf.int64),
        'section_image_in_wit': tf.io.VarLenFeature(dtype=tf.int64),
        'section_contains_table_or_list': tf.io.VarLenFeature(dtype=tf.int64),
        'section_image_captions': tf.io.VarLenFeature(dtype=tf.string),
        'section_image_alt_text': tf.io.VarLenFeature(dtype=tf.string),
        'section_image_raw_attr_desc': tf.io.VarLenFeature(dtype=tf.string),
        'section_image_clean_attr_desc': tf.io.VarLenFeature(dtype=tf.string),
        'section_image_raw_ref_desc': tf.io.VarLenFeature(dtype=tf.string),
        'section_image_clean_ref_desc': tf.io.VarLenFeature(dtype=tf.string),
        'section_contains_images': tf.io.VarLenFeature(dtype=tf.int64)
    }

    def _parse_function(example_proto):
      return tf.io.parse_single_sequence_example(example_proto,
                                                 context_feature_description,
                                                 sequence_feature_description)

    suffix = '.tfrecord*'

    data_path = glob.glob(self.path + self.filepath + suffix)
    raw_dataset = tf.data.TFRecordDataset(data_path, compression_type='GZIP')
    parsed_dataset = raw_dataset.map(_parse_function)

    for d in parsed_dataset:
      split = d[0]['split'].numpy().decode()
      self.data[split].append(d)

if __name__ == "__main__":
  get_sec_and_img_urls(args)

