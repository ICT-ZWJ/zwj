---
dataset_info:
- config_name: caption_matching
  features:
  - name: video_id
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: dim
    dtype: string
  splits:
  - name: test
    num_bytes: 407158
    num_examples: 1503
  download_size: 81730
  dataset_size: 407158
- config_name: captioning
  features:
  - name: video_id
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: dim
    dtype: string
  - name: mc_question
    dtype: string
  - name: mc_answer
    dtype: string
  splits:
  - name: test
    num_bytes: 1725953
    num_examples: 2004
  download_size: 173165
  dataset_size: 1725953
- config_name: multi-choice
  features:
  - name: video_id
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: dim
    dtype: string
  splits:
  - name: test
    num_bytes: 317041
    num_examples: 1580
  download_size: 87086
  dataset_size: 317041
- config_name: yes_no
  features:
  - name: video_id
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: dim
    dtype: string
  splits:
  - name: test
    num_bytes: 236486
    num_examples: 2453
  download_size: 57019
  dataset_size: 236486
configs:
- config_name: caption_matching
  data_files:
  - split: test
    path: caption_matching/test-*
- config_name: captioning
  data_files:
  - split: test
    path: captioning/test-*
- config_name: multi-choice
  data_files:
  - split: test
    path: multi-choice/test-*
- config_name: yes_no
  data_files:
  - split: test
    path: yes_no/test-*
---
