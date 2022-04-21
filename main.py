import os
import yaml
import argparse
import numpy as np
from lce import LCEClassifier
from sklearn.preprocessing import LabelEncoder

from utils.helpers import create_logger, save_experiment
from utils.data_loading import import_data
from utils.results_export import export_results, get_mts_region


if __name__ == '__main__':

    # Load configuration
    parser = argparse.ArgumentParser(description='XEM')
    parser.add_argument('-c', '--config', default='configuration/config.yml', help='Configuration File')
    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        configuration = yaml.safe_load(config_file)
    
    # Create experiment folder
    xp_dir = './results/' + str(configuration['dataset']) + '/window_' + str(int(configuration['window']*100)) + '/xp_' + str(configuration['experiment_run']) + '/'
    save_experiment(xp_dir, args.config)
    log, logclose = create_logger(log_filename=os.path.join(xp_dir, 'experiment.log'))
    
    # Load dataset   
    X_train, y_train, X_validation, y_validation, X_test, y_test = import_data(configuration['dataset'], 
                                                                               configuration['window'], 
                                                                               xp_dir, 
                                                                               configuration['validation_split'],
                                                                               log)
    
    # Fit label encoder
    encoder = LabelEncoder()
    encoder.fit(np.concatenate((y_train, y_validation, y_test), axis=0))
    
    # Fit LCE model - documentation: https://lce.readthedocs.io/en/latest/generated/lce.LCEClassifier.html
    clf = LCEClassifier(n_estimators=configuration['trees'], 
                        max_depth=configuration['max_depth'], 
                        max_samples=configuration['max_samples'],
                        n_jobs=configuration['n_jobs'], 
                        random_state=configuration['random_state'])
    clf.fit(X_train[:,2:], y_train)
    
    # Export results
    results_export, results_train_mts, results_validation_mts, results_test_mts = export_results(clf, encoder,
                                                                                                 configuration, 
                                                                                                 X_train, y_train, 
                                                                                                 X_validation, y_validation, 
                                                                                                 X_test, y_test, 
                                                                                                 xp_dir, log)
    
    # Example of identification of the time window used to classify the first MTS of the test set
    mts_id = 1
    get_mts_region(clf, encoder, X_test, y_test, results_test_mts, mts_id, configuration['window'])
        
    logclose()