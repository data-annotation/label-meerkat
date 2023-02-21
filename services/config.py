import os


project_base_path = 'project'
label_base_path = 'label'

model_path = 'model_data'
rationale_model_path = os.path.join(model_path, 'rationale_model')
rationale_model_data_path = os.path.join(model_path, 'rationale_model_data')

prediction_model_path = os.path.join(model_path, 'prediction_model')
prediction_model_data_path = os.path.join(model_path, 'prediction_model_data')


SentenceTransformer_PATH = "all-MiniLM-L6-v2"


# base_model = 'google/t5-efficient-tiny'
# base_model = 't5-base'
# base_model = 't5-small'
# base_model = 'google/t5-efficient-mini'
base_model = 'valhalla/t5-small-qa-qg-hl'

# model hyper parameters

num_iter = 1
num_data_per_batch = 10
num_epochs_rg = 2
num_epochs_p = 2
learning_rate = 1e-4
per_device_batch_size = 2
