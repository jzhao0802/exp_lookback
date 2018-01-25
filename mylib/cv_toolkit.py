#These functions have been kindly provided by Shaun
import itertools
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd
import collections


def init_pred_dict():
    splits = ['test', 'train']
    predictions = ['probs', 'scores', 'y_pred', 'y_true']

    return {
        split: {
            pred :[] for pred in predictions
        } for split in splits
    }

def flatten_cv_outputs(cv_output):
    #print(list(cv_output['train'].keys()))
    flat_cv_outputs = {
        split: {
            key: list(itertools.chain.from_iterable(value)) if key != 'scores' else value
            for key,value in split_preds.items()
        } for split, split_preds in cv_output.items()

    }
    return flat_cv_outputs


def run_CV(X, y, clf_class, cv_method, params={}, metrics=[], statsmodel=False, flatten=False, return_train_preds=False, grid_search=None):
    #CV_splits = kf.split(df_cleaned, label)
    #Note - if inner_cv_method given, assume this is a nested CV grid search...
    cv_outputs = {
        'models': [],
        'scores': [],
        'predictions': init_pred_dict(),
        'fold_metrics': {
            'test': [],
            'train': []
        }
    }
    #cv_outputs.update({str(func):[] for func in scoring_functions})
    i = 1
    for train_indicies, test_indicies in cv_method.split(X, y):
        print("Running fold ", i, " ...")
        i = i + 1
        if statsmodel:
            clf = clf_class(y.reindex(train_indicies), X.reindex(train_indicies))
            clf = clf.fit()
        else:
            if grid_search:
                grid_search.fit(X.reindex(index=train_indicies, copy=False), y.reindex(index=train_indicies, copy=False))
                if 'best_params' in cv_outputs:
                    cv_outputs['best_params'].append(grid_search.best_params_)
                else:
                    cv_outputs['best_params'] = [grid_search.best_params_]
                clf = grid_search.best_estimator_
            else:
                clf = clf_class(**params)
            clf.fit(X.reindex(index=train_indicies, copy=False), y.reindex(index=train_indicies, copy=False))
            #print(clf)

        #Get outputs for CV
        cv_outputs['models'].append(clf)
        if not statsmodel:
            if return_train_preds:
                cv_outputs['predictions']['train']['probs'].append([x[1] for x in clf.predict_proba(X.loc[train_indicies])])
                cv_outputs['predictions']['train']['scores'].append(clf.score(X.loc[train_indicies], y.loc[train_indicies]))
                cv_outputs['predictions']['train']['y_pred'].append(clf.predict(X.reindex(train_indicies)))
            cv_outputs['predictions']['test']['probs'].append([x[1] for x in clf.predict_proba(X.loc[test_indicies])])
            cv_outputs['predictions']['test']['scores'].append(clf.score(X.loc[test_indicies], y.loc[test_indicies]))
            cv_outputs['predictions']['test']['y_pred'].append(clf.predict(X.reindex(test_indicies)))
        else:
            cv_outputs['predictions']['test']['probs'].append(clf.predict(X.reindex(test_indicies)))
            cv_outputs['predictions']['test']['y_pred'].append([1 if x >= 0.5 else 0 for x in cv_outputs['predictions']['test']['probs'][-1]])
            if return_train_preds:
                cv_outputs['predictions']['train']['probs'].append(clf.predict(X.reindex(train_indicies)))
                cv_outputs['predictions']['train']['y_pred'].append([1 if x > 0.5 else 0 for x in cv_outputs['predictions']['train']['probs'][-1]])

        cv_outputs['predictions']['test']['y_true'].append(y.loc[test_indicies])
        cv_outputs['predictions']['train']['y_true'].append(y.loc[train_indicies])

        cv_outputs['fold_metrics']['test'].append(calc_CV_metrics(**(cv_outputs['predictions']['test']), metrics=metrics))
        if return_train_preds:
            cv_outputs['fold_metrics']['train'].append(calc_CV_metrics(**(cv_outputs['predictions']['train']), metrics=metrics))


    if flatten:
        cv_outputs['predictions'] = flatten_cv_outputs(cv_outputs['predictions'])

    return cv_outputs

def classifaction_report_to_df(report):
    #print(report)
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        #print(row_data)
        row['class'] = row_data[1].strip()
        row['precision'] = float(row_data[2])
        row['recall'] = float(row_data[3])
        row['f1_score'] = float(row_data[4])
        row['support'] = float(row_data[5])
        report_data.append(row)
    return pd.DataFrame.from_dict(report_data)

def calc_CV_metrics(y_true=[], probs=[], y_pred=[], scores=[], models=[], metrics=[], fold_metrics=[], feature_names=[]):
    #print (probs)
    outputs = {}
    for metric in metrics:
        if metric.__name__ == 'precision_recall_curve':
            precision,recall,threshold = metric(y_true, probs)
            outputs[metric.__name__] = {
                'precision': precision,
                'recall': recall,
                'threshold': threshold
            }
            outputs[metric.__name__]['threshold'] = np.insert(outputs[metric.__name__]['threshold'], 0, 0)
        elif metric.__name__ == 'roc_curve':
            outputs[metric.__name__] = metric(y_true, probs)
            outputs[metric.__name__] = {
                'false_pos_rate': outputs[metric.__name__] [0],
                'true_pos_rate': outputs[metric.__name__] [1],
                'threshold': outputs[metric.__name__] [2]
            }
            #outputs[metric.__name__]['threshold'] = np.insert(outputs[metric.__name__]['threshold'], 0, 1)
        elif 'model_sum' in metric.__name__ :
            outputs[metric.__name__] = metric(models, feature_names)
        else:
            outputs[metric.__name__] = metric(y_true, y_pred)
            if 'report' in metric.__name__:
                outputs[metric.__name__] = classifaction_report_to_df(outputs[metric.__name__])


    return outputs

def add_metrics_to_spreadsheet(spreadsheet, model_metrics):
    df_metrics = None
    if not (isinstance(model_metrics, dict)): model_metrics = {'': model_metrics}
    for model,metrics in model_metrics.items():
        model = '_%s'%model
        standard_metrics = {'metric': [], model: []}
        for metric,value in metrics.items():
            if 'curve' in metric:
                df = pd.DataFrame.from_dict(value, orient='columns')
                df.to_excel(spreadsheet, '%s_curve%s' % (metric.split('_')[0], model), index=False)
            elif 'score' in metric:
                standard_metrics['metric'].append(metric)
                standard_metrics[model].append(value)
            else:
                if not isinstance(value, pd.DataFrame): value = pd.DataFrame(value)
                value.to_excel(spreadsheet, '%s%s'%(metric.replace('classification', 'class').replace('confusion', 'conf'), model), index=False)

        if df_metrics is None:
            df_metrics = pd.DataFrame.from_dict(standard_metrics)
        else:
            df_metrics = pd.merge(df_metrics, pd.DataFrame.from_dict(standard_metrics), on='metric')
    df_metrics.to_excel(spreadsheet, 'metric_comparison', index=False)

def calc_imp_feature(cv_outputs, f_cols):
    feature_imp = pd.DataFrame()
    feature_imp['Feature'] = f_cols
    for i in range(len(cv_outputs['models'])):
        col = 'Fold_' + str(i)
        feature_imp[col] = cv_outputs['models'][i].feature_importances_
    feature_imp['Mean_score'] = feature_imp.iloc[:,1:].mean(axis=1)
    feature_imp.sort_values(by= 'Mean_score', axis = 0, ascending = False, inplace =True)
    feature_imp = feature_imp.reset_index(drop = True)
    return feature_imp