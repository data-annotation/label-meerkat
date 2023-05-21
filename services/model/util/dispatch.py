from services.const import TaskType
from services.model.AL import one_training_iteration









def training(task_type: TaskType,
             data: Union[list, dict],
             data_columns: list,
             label_column: str,
             explanation_column: str = None,
             model_id: str = 'test_model',
             old_model_id: str = None):
    if task_type == TaskType.esnli:
        one_training_iteration(data,
                               column_1=data_columns[0],
                               column_2=data_columns[1],
                               explanation_column=explanation_column,
                               labels=['entailment', 'neutral', 'contradiction'],
                               model_id=model_id,
                               old_model_id=old_model_id)
    elif task_type == TaskType.classification:

