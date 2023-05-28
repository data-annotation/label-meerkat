import openai
import pandas as pd
import os

openai.api_key = os.getenv('openai_api_key')

system_prompt = (
  "You are a robot used to identify the sentences classification."
  "Given several sentences separated by '\n', "
  "each sentence start with a number that is sentence index and then follow one sentence separated by '|'"
  "you need to select the classification for each sentence from '%s'"
)

demonstration_question = [
  "1 | I love dogs!",
  "2 | Lock your door, and put the key in your pocket",
  "3 | This movie is amazing!",
  "4 | This traffic is driving me crazy",
  "5 | The food at this restaurant is delicious.",
  "6 | I am so tired today",
  "7 | I had a great time at the party.",
  "8 | the car is flying in the sky",
  "9 | Through the small windows the rooms could be seen",
]

demonstration_answer = [
  "1 | positive",
  "2 | neutral",
  "3 | positive",
  "4 | negative",
  "5 | positive",
  "6 | negative",
  "7 | positive",
  "8 | neutral",
  "9 | neutral",
]


def process_predict_result(results: list, spliter: str = '|', columns: list = None):
  columns = columns or ['id', 'label']
  res = []
  for r in results:
    r = r.strip()
    temp_d = {}
    for k, v in zip(columns, r.split(spliter)):
      v = v.strip()
      temp_d[k] = int(v) if k == 'id' else v
    res.append(temp_d)
  return res


def predict(data: [pd.DataFrame, list],
            prompt: str = system_prompt,
            demonstration_q: list = None,
            demonstration_a: list = None,
            label_list: list = None,
            batch_size: int = 50,
            column_for_predict: str = 'sentence',
            id_column: str = 'id',
            iteration_num: int = None,
            **kwargs):
  iteration_num = iteration_num or len(data)//batch_size+1
  demonstration_q = demonstration_q or demonstration_question
  demonstration_a = demonstration_a or demonstration_answer
  label_list = label_list or ['positive', 'neutral', 'negative']
  prompt_for_redict = prompt.format(','.join(label_list))

  gpt_result = []

  for i in range(iteration_num):
    select_for_gpt_predict = data[i * batch_size:(i + 1) * batch_size]
    if isinstance(data, pd.DataFrame):
      select_for_gpt_predict = select_for_gpt_predict.to_dict('records')
    if not select_for_gpt_predict:
      break
    question = ' \n '.join([' | '.join([str(r[id_column]), r[column_for_predict]])
                            for r in select_for_gpt_predict])

    print(f'{i}/{iteration_num}', len(question.split('\n')))
    res = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": prompt_for_redict},
            {"role": "user", "content": ' \n '.join(demonstration_q)},
            {"role": "assistant", "content": ' \n '.join(demonstration_a)},
            {"role": "user", "content": question}
          ]
    )
    res = res.choices[0].message.content.split('\n')
    print('chatgpt response: ', len(res))
    gpt_result.extend(process_predict_result(res, columns=[id_column, column_for_predict]))

  return gpt_result


if __name__ == "__main__":
  data_for_predict = pd.DataFrame([{'id': 0, 'sentence': 'I love dogs!'},
                                   {'id': 1, 'sentence': 'The weather is terrible today.'},
                                   {'id': 2, 'sentence': 'This movie is amazing!'},
                                   {'id': 3, 'sentence': 'I hate being late.'},
                                   {'id': 4, 'sentence': 'The food at this restaurant is delicious.'},
                                   {'id': 5, 'sentence': 'I am so tired today.'},
                                   {'id': 6, 'sentence': 'I had a great time at the party.'},
                                   {'id': 7, 'sentence': "I can't stand my boss."},
                                   {'id': 8, 'sentence': 'The view from this mountain is breathtaking.'},
                                   {'id': 9, 'sentence': 'I hate when I forget my phone at home.'},
                                   {'id': 10, 'sentence': 'I am grateful for my friends.'},
                                   {'id': 11, 'sentence': 'This traffic is driving me crazy.'},
                                   {'id': 12, 'sentence': 'I love spending time with my family.'},
                                   {'id': 13, 'sentence': "I can't believe how expensive this dress is."},
                                   {'id': 14, 'sentence': 'I enjoy going for a walk in the park.'},
                                   {'id': 15, 'sentence': 'I am so annoyed with my roommate.'},
                                   {'id': 16, 'sentence': 'This book is inspiring.'},
                                   {'id': 17, 'sentence': 'I am so stressed with work.'},
                                   {'id': 18, 'sentence': 'I love playing sports.'},
                                   {'id': 19, 'sentence': 'This music is terrible.'}])
  print(predict(data_for_predict))
