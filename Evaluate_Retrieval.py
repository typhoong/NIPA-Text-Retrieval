'''
2022_NIPA task06 text retrieval
columns : 'question_id', 'paragraph_id', 'public'
Mean Reciprocal Rank (ln 14: k)
'''


import sys
import pandas as pd

k = 10

def load_result(path, pred=False):
    try:
        result = pd.read_csv(path)
        # result = result.sort_values(by='question_id')

        if pred is False: # answer
            p_type_li = result['public'].tolist()
        else: # prediction
            p_type_li = None

        label = list(result['paragraph_id'])

        return list(result['question_id']), label, p_type_li

    except Exception as e:
        assert False, e


def evaluate(label, prediction):
    # label = list(map(str, label))
    # prediction = list(map(str, prediction))
    
    prediction = [a.split(',') if len(a.split(','))<k else a.split(',')[:k] for a in prediction]
    
    total = 0
    
    for i,ans in enumerate(label):
        if ans in prediction[i]:
            # print(prediction[i].index(ans))
            total += 1/(1+prediction[i].index(ans))
    
    return total/len(label)


def mrr(answer, pred):

    a_id, a_answer, p_type_li = load_result(answer)
    p_id, p_pred, _ = load_result(pred, pred=True)

    assert len(a_id) == len(p_id), 'The number of predictions and answers are not the same'
    assert set(p_id) == set(a_id), 'The prediction ids and answer ids are not the same'
    assert a_id == p_id, 'Please match the order with the sample submission.'

    pub_a_id, pub_answer, prv_a_id, prv_answer = [], [], [], []
    pub_p_id, pub_pred, prv_p_id, prv_pred = [], [], [], []

    for idx, t in enumerate(p_type_li):
        if t:
            pub_a_id.append(a_id[idx])
            pub_answer.append(a_answer[idx])
            pub_p_id.append(p_id[idx])
            pub_pred.append(p_pred[idx])
        else:
            prv_a_id.append(a_id[idx])
            prv_answer.append(a_answer[idx])
            prv_p_id.append(p_id[idx])
            prv_pred.append(p_pred[idx])

    # sort
    pub_ans = pd.DataFrame({'question_id': pub_a_id, 'paragraph_id': pub_answer}).sort_values('question_id', ignore_index=True)
    pub_pred = pd.DataFrame({'question_id': pub_p_id, 'paragraph_id': pub_pred}).sort_values('question_id', ignore_index=True)
    prv_ans = pd.DataFrame({'question_id': prv_a_id, 'paragraph_id': prv_answer}).sort_values('question_id', ignore_index=True)
    prv_pred = pd.DataFrame({'question_id': prv_p_id, 'paragraph_id': prv_pred}).sort_values('question_id', ignore_index=True)
    
    # mrr
    score = evaluate(prediction=pub_pred['paragraph_id'], label=pub_ans['paragraph_id'])
    pScore = evaluate(prediction=prv_pred['paragraph_id'], label=prv_ans['paragraph_id'])

    return score, pScore


if __name__ == '__main__':

    answer = sys.argv[1]
    pred = sys.argv[2]

    try:
        import time
        start = time.time()
        score, pScore = mrr(answer, pred)
        print(f'score={score},pScore={pScore}')
        print(f'Elapsed Time: {time.time() - start}')

    except Exception as e:
        print(f'evaluation exception error: {e}', file=sys.stderr)
        sys.exit()
