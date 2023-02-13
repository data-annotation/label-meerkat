from services.model.predictor import predict
from services.model.util.model_input import prediction_model_preprocessing


def one_iter():
  predicted_rationale = predict(rationale_dataset_test_dataset,
                                'rationale',
                                TEMP_FOLDER_PATH + 'rationale_model_' + str(index),
                                'ignore')
  prediction_model_test_dataset = test_dataset.add_column("generated_rationale", predicted_rationale)
  prediction_model_test_dataset = prediction_model_preprocessing(prediction_model_test_dataset)

  predicted_prediction = predict(prediction_model_test_dataset,
                                 'prediction',
                                 TEMP_FOLDER_PATH + 'prediction_model_' + str(index),
                                 criteria)