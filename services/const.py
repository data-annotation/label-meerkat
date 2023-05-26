from enum import Enum


class ModelStatus(Enum):
    running = 1,
    free = 0


class TrainingWay(str, Enum):
    new = 'new'
    continued = 'continued'


class GetUnlabeledWay(str, Enum):
    random = 'random'
    similarity = 'similarity'


class TaskType(str, Enum):
    classification = 'classification'
    sequence_tag = 'sequence_tag'
    relation = 'relation'
    esnli = 'esnli'


class ConfigName(str, Enum):
    esnli = 'esnli'
    fairy_tale = 'fairy_tale'


class ModelForTrain(str, Enum):
  Bert = 'Bert'
  Bart = 'Bart'
  GPT2 = 'BPT2'


label_schema = {
    # 'Time flies like an arrow; fruit flies like a banana.'
    TaskType.sequence_tag: [{'s': 0, 'e': 4, 'label': 'aaa'},
                            {'s': 19, 'e': 24, 'label': 'bbb'}],
    TaskType.classification: 'aaa',
    TaskType.relation: 'bbb',
}

CONFIG_NAME_MAPPING = {
    ConfigName.esnli: ConfigName.esnli,
    ConfigName.fairy_tale: TaskType.classification
}

CONFIG_MAPPING = {
    ConfigName.esnli: {'columns': ['sentence1', 'sentence2', 'id'],
                       'data_columns': ['sentence1', 'sentence2'],
                       'id_columns': ['id'],
                       'task_type': TaskType.relation.value,
                       'default_label_config': {'labels': ['entailment',
                                                           'neutral',
                                                           'contradiction'],
                                                'columns': ['label', 'id', 'explanation'],
                                                'label_column': 'label',
                                                'id_columns': ['id']}},
    ConfigName.fairy_tale: {'columns': ['text', 'document_id'],
                            'data_columns': ['text'],
                            'task_type': TaskType.classification.value,
                            'text_name_column': 'document_id',
                            'default_label_config': {'labels': ['positive',
                                                                'neutral'
                                                                'negative'],
                                                     'columns': ['label', 'id'],
                                                     'label_column': 'label',
                                                     'id_columns': ['id']}},
    TaskType.sequence_tag: {'columns': ['sentence', 'id'],
                            'data_columns': ['sentence'],
                            'id_columns': ['id'],
                            'task_type': TaskType.sequence_tag.value,
                            'default_label_config': {'columns': ['label', 'id'],
                                                     'label_column': 'label',
                                                     'labels': ['aaa', 'bbb', 'ccc'],
                                                     'id_column': 'id'}},

    TaskType.classification: {'columns': ['sentence'],
                              'data_columns': ['sentence'],
                              'task_type': TaskType.classification.value,
                              'default_label_config': {'columns': ['label', 'id'],
                                                       'label_column': 'label',
                                                       'labels': ['aaa', 'bbb', 'ccc'],
                                                       'id_column': 'id'}},

    TaskType.relation: {'columns': ['sentence1', 'sentence2', 'id1', 'id2'],
                        'data_columns': ['sentence1', 'sentence2'],
                        'id_columns': ['id1', 'id2'],
                        'task_type': TaskType.relation.value,
                        'default_label_config': {'columns': ['label', 'id1', 'id2'],
                                                 'label_columns': ['label'],
                                                 'labels': ['aaa', 'bbb', 'ccc'],
                                                 'id_columns': ['id']}},
  }



# classification
# {
#     "columns": [
#         "text", "document_id"
#     ],
#     "data_columns": [
#         "sentence"
#     ],
#     "text_name_column": "document_id"
#     "task_type": "classification",
#     "default_label_config": {
#         "columns": [
#             "label",
#             "id"
#         ],
#         "label_column": "label",
#         "labels": [
#             "positive",
#             "negative"
#         ],
#         "id_column": "id"
#     }
# }