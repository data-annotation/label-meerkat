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

