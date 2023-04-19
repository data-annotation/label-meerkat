from fastapi.testclient import TestClient
from services.main import app
import sys
sys.path.append('/home/xh9487963/projects/label-meerkat/')

client = TestClient(app)


def test_project_main():
    response = client.get("/projects")
    assert response.status_code == 200
    assert response.json() == [
        {
            "id": 1,
            "name": "test1",
            "create_time": "2023-04-18T15:04:01",
            "update_time": "2023-04-18T15:04:01",
            "labels": [
                {
                    "label_id": 1,
                    "user_id": 1,
                    "config": {
                        "choice": [
                            "entailment",
                            "neutral",
                            "contradiction"
                        ],
                        "columns": [
                            "label",
                            "id",
                            "explanation"
                        ],
                        "label_column": "label",
                        "label_data_type": "index",
                        "id_column": "id"
                    },
                    "current_model": "76f54133e6084e20870d0a809380a57c",
                    "create_time": "2023-04-18 15:04:01",
                    "update_time": "2023-04-18 16:24:09"
                }
            ]
        }
    ]


def test_get_single_project():
    response = client.get("/projects/1")
    assert response.status_code == 200
    assert response.json() == {
        "project_meta": {
            "id": 1,
            "name": "test1",
            "file_path": "e264af72a81149c8b610143ef6562f12",
            "config": {
                "columns": [
                    "sentence1",
                    "sentence2",
                    "id"
                ],
                "default_label_config": {
                    "choice": [
                        "entailment",
                        "neutral",
                        "contradiction"
                    ],
                    "columns": [
                        "label",
                        "id",
                        "explanation"
                    ],
                    "label_column": "label",
                    "label_data_type": "index",
                    "id_column": "id"
                }
            },
            "create_time": "2023-04-18T15:04:01",
            "update_time": "2023-04-18T15:04:01"
        },
        "data_num": 0,
        "label_num": 0
    }


def test_get_label():
    response = client.get("/projects/1/labels")
    assert response.status_code == 200
    assert response.json() == [
        {
            "id": 1,
            "name": "test1_label",
            "user_id": 1,
            "project_id": 1,
            "config": {
                "choice": [
                    "entailment",
                    "neutral",
                    "contradiction"
                ],
                "columns": [
                    "label",
                    "id",
                    "explanation"
                ],
                "label_column": "label",
                "label_data_type": "index",
                "id_column": "id"
            },
            "create_time": "2023-04-18T15:04:01",
            "update_time": "2023-04-18T16:24:09"
        }
    ]

# def test_create_new():
#     response = client.post('/projects', 
#                            json={
#                                 'token' : 'abc',
#                                 'name' : 'abc',

#                            })

if __name__ == '__main__':
    test_project_main()
