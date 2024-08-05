import os
import re
import csv
import json
import random
import datetime
import holidays

from paddle.io import Dataset


class EvalDatasets(Dataset):

    def __init__(self, data_root, dataset_name):
        super().__init__()
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.data = self.read_data()

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def read_data(self):
        raise NotImplementedError

    def check_answer(self, outputs, answer):
        raise NotImplementedError


class MathBenchmarks(EvalDatasets):
    dataset_names = {
        'asdiv': 'cv_asdiv-a',
        'mawps': 'cv_mawps',
        'svamp': 'cv_svamp_augmented'
    }

    def __init__(self, data_root, dataset_name, split='train_dev'):
        data_root = os.path.join(data_root, 'svamp')
        dataset_name = self.dataset_names[dataset_name]
        self.split = split
        super().__init__(data_root, dataset_name)

    @staticmethod
    def _read_data_from_csv(data_file):
        data = []
        with open(data_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == 'Question' and row[1] == 'Numbers':
                    continue
                question = row[0]
                numbers = row[1].split(' ')
                # replace 'number0', 'number1', ... with actual numbers
                for i, number in enumerate(numbers):
                    question = question.replace(f'number{i}', number)
                answer = row[3]
                data.append((question, answer))
        return data

    def read_data(self):
        if self.split == 'train' or self.split == 'dev':
            data_file = os.path.join(self.data_root, self.dataset_name, 'fold0', f'{self.split}.csv')
            return self._read_data_from_csv(data_file)
        elif self.split == 'train_dev':
            data_file = os.path.join(self.data_root, self.dataset_name, 'fold0', 'train.csv')
            data = self._read_data_from_csv(data_file)
            data_file = os.path.join(self.data_root, self.dataset_name, 'fold0', 'dev.csv')
            data.extend(self._read_data_from_csv(data_file))
            return data

    def check_answer(self, outputs, label):
        # Find number in label
        match = re.search(r'\d+\.?\d*', label)
        if not match:
            return False
        label = float(match.group())
        # Check whether the output contains calculator api call
        expression_match, api_result = re.search(r'Calculator\((.*)\)', outputs), None
        if expression_match:
            match = re.search(r'\d+\.?\d*', outputs[outputs.find(')') + 1:])
            api_result = float(match.group()) if match else None
            outputs = outputs[outputs.find(']') + 1:] if ']' in outputs else outputs
        # Check original result
        match = re.search(r'\d+\.?\d*', outputs)
        first_number = float(match.group()) if match else None
        match = re.search(r'=\s*(\d+\.?\d*)', outputs)
        first_number_after_equal = float(match.group(1)) if match else None
        ori_result = first_number if first_number_after_equal is None else first_number_after_equal
        return label == api_result or label == ori_result


class QABenchmarks(EvalDatasets):
    dataset_names = {
        'squad': 'Squad',
        'google_re': 'Google_RE',
        'trex': 'TREx',
        'concept_net': 'ConceptNet'
    }

    def __init__(self, data_root, dataset_name):
        data_root = os.path.join(data_root, 'lama')
        super().__init__(data_root, self.dataset_names[dataset_name])

    def read_data(self):
        data_path = os.path.join(self.data_root, self.dataset_name)
        data = []
        for file_name in os.listdir(str(data_path)):
            if file_name.endswith('.jsonl'):
                data_file = os.path.join(str(data_path), file_name)
                with open(data_file, 'r') as f:
                    for line in f:
                        content = json.loads(line)
                        if self.dataset_name in ['Squad', 'Google_RE', 'ConceptNet']:
                            sentence = content['masked_sentences'][0]
                        elif self.dataset_name in ['TREx']:
                            sentence = content['evidences'][0]['masked_sentence']
                        match = re.search(r'.*\[MASK]\s*[.,!?]*\s*$', sentence)
                        if match:
                            sentence = ('Please complete the following text (only return the missing part)'
                                        ' so that it is factually correct: ') + \
                                       sentence[:sentence.find('[MASK]')]
                            data.append((sentence, content['obj_label']))
        return data

    def check_answer(self, outputs, label):
        qa_match, api_check = re.search(r'QA\((.*)\)', outputs), False
        if qa_match:
            front_idx, end_idx = outputs.find(')'), outputs.find(']')
            if front_idx != -1 and end_idx != -1 and end_idx > front_idx:
                api_outputs = outputs[front_idx: end_idx + 1]
                api_check = label.lower() in api_outputs.lower()
                outputs = api_outputs[end_idx + 1:]
        outputs_head = outputs.lower().split(' ')[: 5]
        ori_check = False
        for o in outputs_head:
            if label.lower() in o:
                ori_check = True
        return api_check or ori_check


class SearchQA(EvalDatasets):
    dataset_names = {
        'web_qs': 'WebQS',
        'nq': 'NQ',
        'trivia_qa': 'TriviaQA',
    }
    file_names = {
        'WebQS': 'test.json',
        'NQ': 'NQ-open.efficientqa.test.1.1.jsonl',
        'TriviaQA': 'qa/verified-wikipedia-dev.json'
    }

    def __init__(self, data_root, dataset_name):
        data_root = os.path.join(data_root, 'lama')
        super().__init__(data_root, self.dataset_names[dataset_name])

    def read_data(self):
        data_path = os.path.join(self.data_root, self.dataset_name)
        data_file = open(os.path.join(str(data_path), self.file_names[self.dataset_name]), 'r')
        data = []
        if self.dataset_name == 'WebQS':
            raw_data = json.load(data_file)
            for content in raw_data:
                question = content['utterance']
                answers = content['targetValue']
                pattern = re.compile(r'\(description (.*?)\)')
                ans_list = pattern.findall(answers)
                ans_list = [ans.replace('\"', '') for ans in ans_list]
                data.append(('Answer the following question: ' + question, ans_list))
        elif self.dataset_name == 'NQ':
            for line in data_file:
                content = json.loads(line)
                data.append(('Answer the following question: ' + content['question'] + ' ?', content['answer']))
        elif self.dataset_name == 'TriviaQA':
            raw_data = json.load(data_file)['Data']
            for content in raw_data:
                question = content['Question']
                answer = content['Answer']['NormalizedAliases']
                data.append(('Answer the following question: ' + question, answer))
            data_file = open(os.path.join(str(data_path), 'qa/verified-web-dev.json'), 'r')
            raw_data = json.load(data_file)['Data']
            for content in raw_data:
                question = content['Question']
                answer = content['Answer']['NormalizedAliases']
                data.append(('Answer the following question: ' + question, answer))
        return data

    @staticmethod
    def check_ans_list(text, ans_list):
        for ans in ans_list:
            if ans.lower() in text.lower():
                return True
        return False

    def check_answer(self, outputs, answer):
        search_match, api_check = re.search(r'QA\((.*)\)', outputs), False
        if search_match:
            front_idx, end_idx = outputs.find(')'), outputs.find(']')
            if front_idx != -1 and end_idx != -1 and end_idx > front_idx:
                api_outputs = outputs[front_idx: end_idx + 1]
                api_check = self.check_ans_list(api_outputs, answer)
                outputs = api_outputs[end_idx + 1:]
        outputs_head = outputs.lower().split(' ')[: 20]
        outputs_head_str = ' '.join(outputs_head)
        ori_check = self.check_ans_list(outputs_head_str, answer)
        return api_check or ori_check


class MultiLanguageQA(EvalDatasets):

    def __init__(self, data_root, dataset_name, split='test'):
        self.split = split
        self.language = dataset_name.split('-')[-1]
        super().__init__(data_root, dataset_name)

    def read_data(self):
        data_path = os.path.join(self.data_root, 'MLQA_V1', self.split)
        data = []
        for file_name in os.listdir(str(data_path)):
            if file_name.endswith('.json') and 'context-en' in file_name and 'question-' + self.language in file_name:
                data_file = os.path.join(str(data_path), file_name)
                with (open(data_file, 'r') as f):
                    file_data = json.load(f)['data']
                    for samples in file_data:
                        sample_data = samples['paragraphs']
                        for sample in sample_data:
                            content = sample['context']
                            qas = sample['qas'][0]
                            question = qas['question']
                            sentence = 'Your task is to answer a question based on the following paragraph: ' + \
                                       content + ' Now answer the following question in English: ' + question
                            data.append((sentence, qas['answers'][0]['text']))
        return data

    def check_answer(self, outputs, label):
        translator_match, api_check = re.search(r'MT\((.*)\)', outputs), False
        if translator_match:
            front_idx, end_idx = outputs.find(')'), outputs.find(']')
            if front_idx != -1 and end_idx != -1 and end_idx > front_idx:
                api_outputs = outputs[front_idx: end_idx + 1]
                api_check = label.lower() in api_outputs.lower()
                outputs = api_outputs[end_idx + 1:]
        outputs_head = outputs.lower().split(' ')[: 10]
        ori_check = False
        for o in outputs_head:
            if label.lower() in o:
                ori_check = True
        return api_check or ori_check


class DateBenchmark(EvalDatasets):

    def __init__(self, data_root, dataset_name, seed=0):
        random.seed(seed)
        self.date_range = 5 * 365
        super().__init__(data_root, dataset_name)

    @staticmethod
    def _current_date():
        return datetime.date.today()

    def _random_past_date(self):
        return self._current_date() - datetime.timedelta(days=random.randint(0, self.date_range))

    def _random_future_date(self):
        return self._current_date() + datetime.timedelta(days=random.randint(0, self.date_range))

    def read_data(self):
        current_date = self._current_date()
        data = []
        for _ in range(200):
            past_date = self._random_past_date()
            future_date = self._random_future_date()
            question = 'How many days ago was ' + str(past_date) + '?'
            data.append((question, str((current_date - past_date).days)))
            question = 'How many days are there until ' + str(future_date) + '?'
            data.append((question, str((future_date - current_date).days)))
        for _ in range(200):
            past_date = self._random_past_date()
            days = (current_date - past_date).days
            question = 'What day of the week was it ' + str(days) + ' days ago?'
            data.append((question, past_date.strftime('%A')))
            question = 'What day of the month was it ' + str(days) + ' days ago?'
            data.append((question, str(past_date.day)))
            question = 'What month was it ' + str(days) + ' days ago?'
            data.append((question, past_date.strftime('%B')))
            question = 'What year was it ' + str(days) + ' days ago?'
            data.append((question, str(past_date.year)))
        for _ in range(200):
            future_date = self._random_future_date()
            days = (future_date - current_date).days
            question = 'What day of the week will it be in ' + str(days) + ' days?'
            data.append((question, future_date.strftime('%A')))
            question = 'What day of the month will it be in ' + str(days) + ' days?'
            data.append((question, str(future_date.day)))
            question = 'What month will it be in ' + str(days) + ' days?'
            data.append((question, future_date.strftime('%B')))
            question = 'What year will it be in ' + str(days) + ' days?'
            data.append((question, str(future_date.year)))
        for _ in range(200):
            past_date = self._random_past_date()
            future_date = self._random_future_date()
            question = 'What day of the week was it on ' + str(past_date) + ' ?'
            data.append((question, past_date.strftime('%A')))
            question = 'What day of the week is it on ' + str(future_date) + ' ?'
            data.append((question, future_date.strftime('%A')))
        for _ in range(200):
            rand_date = self._random_past_date() if random.random() < 0.5 else self._random_future_date()
            day_gaps = [-2, -1, 0, 1, 2]
            gap_strs = ['the day before yesterday', 'yesterday', 'today', 'tomorrow', 'the day after tomorrow']
            for day_gap, gap_str in zip(day_gaps, gap_strs):
                date = rand_date + datetime.timedelta(days=day_gap)
                verb = 'is' if day_gap >= 0 else 'was'
                question = 'Today is ' + str(rand_date) + '. What day of the week ' + verb + ' it ' + gap_str + '?'
                data.append((question, date.strftime('%A')))
                question = 'Today is ' + str(rand_date) + '. What day of the month ' + verb + ' it ' + gap_str + '?'
                data.append((question, str(date.day)))
                question = 'Today is ' + str(rand_date) + '. What month ' + verb + ' it ' + gap_str + '?'
                data.append((question, date.strftime('%B')))
                question = 'Today is ' + str(rand_date) + '. What year ' + verb + ' it ' + gap_str + '?'
                data.append((question, str(date.year)))
        for _ in range(200):
            rand_date = self._random_past_date() if random.random() < 0.5 else self._random_future_date()
            year = rand_date.year
            all_holidays = holidays.country_holidays('US', years=year)
            for date, name in sorted(all_holidays.items()):
                date_obj = datetime.datetime.strptime(str(date), '%Y-%m-%d').date()
                verb = 'is' if date_obj >= rand_date else 'was'
                question = 'Today is ' + str(rand_date) + '. What day of the week ' + verb + ' ' + name + ' this year?'
                data.append((question, date_obj.strftime('%A')))
                question = 'Today is ' + str(rand_date) + '. What day of the month ' + verb + ' ' + name + ' this year?'
                data.append((question, str(date_obj.day)))
                question = 'Today is ' + str(rand_date) + '. What month ' + verb + ' ' + name + ' this year?'
                data.append((question, date_obj.strftime('%B')))
                day_delta = rand_date - date_obj
                verb = 'ago was' if day_delta.days > 0 else 'are there until'
                question = 'Today is ' + str(rand_date) + '. How many days ' + verb + ' ' + name + ' this year?'
                data.append((question, str(abs(day_delta.days))))
        return data

    def check_answer(self, outputs, label):
        calendar_match, api_check = re.search(r'Calendar\((.*)\)', outputs), False
        if calendar_match:
            front_idx, end_idx = outputs.find(')'), outputs.find(']')
            if front_idx != -1 and end_idx != -1 and end_idx > front_idx:
                api_outputs = outputs[front_idx: end_idx + 1]
                api_check = label.lower() in api_outputs.lower()
                outputs = api_outputs[end_idx + 1:]
        outputs_head = outputs.lower().split(' ')[: 5]
        ori_check = False
        for o in outputs_head:
            if label.lower() in o:
                ori_check = True
        return api_check or ori_check
