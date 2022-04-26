import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def get_res_mts(X, y, y_pred, encoder):
    """
    Export the results for a set at MTS level
    
    Parameters
    ----------
    X: array
        Set
        
    y: array
        Labels of the set
        
    y_pred: array
        Model predictions on the set
    
    encoder: object
        Label encoder
        
    Returns
    -------
    results: array
        Results at MTS level
    """
    df_res = pd.concat([pd.DataFrame(X[:,:2]), pd.DataFrame(y), 
                        pd.DataFrame(encoder.transform(y)),
                        pd.DataFrame(y_pred)], axis=1)
    col_names = ['id', 'timestamp', 'target', 'target_num']
    for i in range(0, y_pred.shape[1]):
        col_names = np.append(col_names, str(i))
    df_res.columns = col_names 
    df_res_mts = df_res.groupby(['id']).max().reset_index(drop=False)
    df_res_mts['pred_num'] = df_res_mts.iloc[:, -y_pred.shape[1]:].idxmax(axis=1)
    df_res_mts['max_pred'] = df_res_mts.iloc[:, -y_pred.shape[1]-1:-1].max(axis=1)
    return df_res_mts


def export_results(clf, encoder, configuration, X_train, y_train, X_validation, 
                   y_validation, X_test, y_test, xp_dir, log=print):
    """
    Export the results of the experiment
    
    Parameters
    ----------
    clf: object
        Trained model
        
    encoder: object
        Label encoder
        
    configuration: array
        Elements of the configuration file
        
    X_train: array
        Train set
        
    y_train: array
        Labels of the train set
        
    X_validation: array
        Validation set
        
    y_validation: array
        Labels of the validation set
        
    X_test: array
        Test set
        
    y_test: array
        Labels of the test set
        
    xp_dir: string
        Folder of the experiment
    
    log: string
        Processing of the outputs
    
    Returns
    -------
    results: array
        Results of the experiment
    """    
    # Make train/validation/test predictions
    y_pred_train = clf.predict_proba(X_train[:,2:])
    res_train_mts = get_res_mts(X_train, y_train, y_pred_train, encoder)
    accuracy_train = accuracy_score(res_train_mts.target_num, 
                                    res_train_mts.pred_num.astype(int))
    
    accuracy_validation = '-'
    if configuration['validation_split'][1] != 0:
        y_pred_validation = clf.predict_proba(X_validation[:,2:])
        res_validation_mts = get_res_mts(X_validation, y_validation, 
                                         y_pred_validation, encoder)
        accuracy_validation = accuracy_score(res_validation_mts.target_num, 
                                             res_validation_mts.pred_num.astype(int))
            
    y_pred_test = clf.predict_proba(X_test[:,2:])
    res_test_mts = get_res_mts(X_test, y_test, y_pred_test, encoder)
    accuracy_test = accuracy_score(res_test_mts.target_num, 
                                   res_test_mts.pred_num.astype(int))
    
    log('\nAccuracy train: {0}'.format(accuracy_train))
    log('Accuracy validation: {0}'.format(accuracy_validation))
    log('Accuracy test: {0}'.format(accuracy_test))
    
    # Export results
    results_export = pd.DataFrame([[configuration['dataset'],
                                             configuration['experiment_run'],
                                             int(configuration['window']*100),
                                             configuration['trees'],
                                             configuration['max_depth'],
                                             configuration['validation_split'],
                                             accuracy_train,
                                             accuracy_validation,
                                             accuracy_test]])
    results_export.columns = ['Dataset', 'Experiment', 'Window_Percentage', 'Trees', 
                              'Depth', 'Validation_Split', 'Accuracy_Train',
                              'Accuracy_Validation','Accuracy_Test']
    results_export.to_csv(xp_dir+'/results_export.csv', index=False)
    return results_export, res_train_mts, res_validation_mts, res_test_mts


def get_mts_region(clf, encoder, X, y, res_mts, mts_id, window):
    """
    Get the MTS region used by XEM to predict
    
    Parameters
    ----------
    clf: object
        Trained model
        
    encoder: object
        Label encoder
                
    X: array
        Set
        
    y: array
        Labels of the set
        
    res_mts: array
        Results on the set at MTS level

    mts_id: integer
        ID of the MTS
        
    window: float
        Size of the time window
    """
    pred_num = int(np.array(res_mts.loc[res_mts.id==mts_id, "pred_num"])[0])
    max_pred = np.array(res_mts.loc[res_mts.id==mts_id, "max_pred"])[0]    
    pred_proba = clf.predict_proba(X[:,2:])
    mts_region = X[pred_proba[:,pred_num]==max_pred]
    mts_length = (len(X)/len(np.unique(X[:,0]))-1)/(1-window)
    window_size = int(window*mts_length)
    
    for i in range(0, len(mts_region)):
        if i == 0:
            start = mts_region[0][1]-window_size+1
        if i == (len(mts_region)-1):
            end = mts_region[0][1]
    
    print('\nExample')
    print('MTS ID: {0}'.format(mts_id))
    print('MTS label: {0}'.format(y[X[:,0]==mts_id][0]))
    print('XEM prediction: {0}'.format(encoder.inverse_transform([pred_num])[0]))
    print('MTS region used by XEM to predict: [{0}, {1}]'.format(start, end))