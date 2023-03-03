from fastapi.testclient import TestClient

from services.main import app

client = TestClient(app)


def test_read_main():
  data = {
    "config_data": {
      "label_column": "label",
      "label_choice": ["entailment", "neutral", "contradiction"],
      "explanation_column": "explanation_1"
    },
    "label_data": {
      "label": [1, 2, 0, 1, 0, 2, 2, 0, 1, 1],
      "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
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
  response = client.post("/labels", json=data)
  assert response.status_code == 200
  assert response.json() == {"msg": "Hello World"}

