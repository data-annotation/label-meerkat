import os


project_base_path = 'project'
label_base_path = 'label'
predict_result_path = 'predict_result'

model_path = 'model_data'

SentenceTransformer_PATH = "../all-MiniLM-L6-v2"

sqlite_dsn = 'sqlite:///test.db'
base_model = 'google/t5-efficient-tiny'
# base_model = 't5-base'
# base_model = 't5-small'
# base_model = 'google/t5-efficient-mini'
# base_model = 'valhalla/t5-small-qa-qg-hl'

# model hyper parameters

num_iter = 1
num_data_per_batch = 10
num_epochs_rg = 10
num_epochs_p = 1
learning_rate = 1e-4
per_device_batch_size = 10

max_model_num_for_one_label = 2


available_models = {
  'classification': ['bert', 'bart', 'svm'],
  'relation': ['t5'],
}
