from services.model.AL import one_training_iteration
from services.model.arch import predict_pipeline
from pprint import pprint

"""
{
    "config_data": { 
    "label_column": "label",
    "label_choice": ["entailment", "neutral", "contradiction"],
    "explanation_column": "explanation_1"
  },
    "label_data": {
      "label": [1, 2, 0, 1, 0, 2, 2, 0, 1, 1],
      "id": [0,1,2,3,4,5,6,7,8,9],
      "explanation_1": ["the person is not necessarily training his horse",
      "One cannot be on a jumping horse cannot be a diner ordering food.",
      "a broken down airplane is outdoors",
      "Just because they are smiling and waving at a camera does not imply their parents or anyone is anyone behind it",
      "The children must be present to see them smiling and waving.",
      "One cannot be smiling and frowning at the same time.",
      "One cannot be in the middle of a bridge if they are on the sidewalk.",
      "jumping on skateboard is the same as doing trick on skateboard.",
      "Just because the boy is jumping on a skateboard does not imply he is wearing safety equipment",
      "it is not necessarily true the man drinks his juice"]
    }

}

"""


test_data = {'premise': ['A person on a horse jumps over a broken down airplane.',
                         'A person on a horse jumps over a broken down airplane.',
                         'A person on a horse jumps over a broken down airplane.',
                         'Children smiling and waving at camera',
                         'Children smiling and waving at camera',
                         'Children smiling and waving at camera',
                         'A boy is jumping on skateboard in the middle of a red bridge.',
                         'A boy is jumping on skateboard in the middle of a red bridge.',
                         'A boy is jumping on skateboard in the middle of a red bridge.',
                         'An older man sits with his orange juice at a small table in a coffee shop while employees in bright colored shirts smile in the background.'],
             'hypothesis': ['A person is training his horse for a competition.',
                            'A person is at a diner, ordering an omelette.',
                            'A person is outdoors, on a horse.',
                            'They are smiling at their parents',
                            'There are children present',
                            'The kids are frowning',
                            'The boy skates down the sidewalk.',
                            'The boy does a skateboarding trick.',
                            'The boy is wearing safety equipment.',
                            'An older man drinks his juice as he waits for his daughter to get off work.'],
             'label': [1, 2, 0, 1, 0, 2, 2, 0, 1, 1],
             'explanation_1': ['the person is not necessarily training his horse',
                               'One cannot be on a jumping horse cannot be a diner ordering food.',
                               'a broken down airplane is outdoors',
                               'Just because they are smiling and waving at a camera does not imply their parents or anyone is anyone behind it',
                               'The children must be present to see them smiling and waving.',
                               'One cannot be smiling and frowning at the same time.',
                               'One cannot be in the middle of a bridge if they are on the sidewalk.',
                               'jumping on skateboard is the same as doing trick on skateboard.',
                               'Just because the boy is jumping on a skateboard does not imply he is wearing safety equipment',
                               'it is not necessarily true the man drinks his juice']}

one_training_iteration(test_data, model_id='abc')

r, p = predict_pipeline(test_data,
                        model_id='abc',
                        label_id=1)

pprint([r, p])

