from datasets import load_dataset
import json
import pandas as pd

dataset = load_dataset("thu-coai/esconv")
print(f"Total conversations: {len(dataset['train'])}")

rows = []
for i, raw in enumerate(dataset['train']):
    if i >= 100:
        break

    conv = json.loads(raw['text'])
    survey = conv.get('survey_score', {}).get('seeker', {})

    for turn in conv.get('dialog', []):
        speaker = "SEEKER" if turn['speaker'] == 'usr' else "SUPPORTER"
        rows.append({
            'conv_num': i + 1,
            'experience_type': conv.get('experience_type'),
            'emotion_type': conv.get('emotion_type'),
            'problem_type': conv.get('problem_type'),
            'situation': conv.get('situation', ''),
            'speaker': speaker,
            'strategy': turn.get('strategy', '') if turn['speaker'] == 'sys' else '',
            'message': turn['text'],
            # seeker survey (one value per conversation, repeated for each turn)
            'seeker_survey_empathy': survey.get('empathy', ''),
            'seeker_survey_relevance': survey.get('relevance', ''),
            'seeker_survey_initial_emotion': survey.get('initial_emotion_intensity', ''),
            'seeker_survey_final_emotion': survey.get('final_emotion_intensity', ''),
            # supporter survey
            'supporter_survey_relevance': conv.get('survey_score', {}).get('supporter', {}).get('relevance', ''),
            # extra questions asked after conversation
            'seeker_question1': conv.get('seeker_question1', ''),
            'seeker_question2': conv.get('seeker_question2', ''),
            'supporter_question1': conv.get('supporter_question1', ''),
            'supporter_question2': conv.get('supporter_question2', ''),
        })

df = pd.DataFrame(rows)
df.to_csv('results/dataset_sample_100.csv', index=False)
print("Saved to results/dataset_sample_100.csv")
print(f"Total rows (turns): {len(df)}")
print(f"Columns: {list(df.columns)}")
