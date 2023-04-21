from enum import Enum


class ModelStatus(Enum):
  running = 1,
  free = 0


class GetUnlabeledWay(Enum):
  random = 'random'
  similarity = 'similarity'

