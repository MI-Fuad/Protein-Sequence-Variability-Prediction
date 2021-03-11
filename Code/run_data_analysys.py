from SeqTrajData import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def run_data_analysis_w_crossvalidation(seqtd_obj, method=LinearRegression(), predictor_name='LR'):
    seqtd_obj.create_predictor(method, predictor_name)

    seqtd_obj.run_crossvalidation()

def run_data_analysis(seqtd_obj, method=LinearRegression(), predictor_name='LR'):
    seqtd_obj.holdout()

    seqtd_obj.create_predictor(method, predictor_name)

    seqtd_obj.run_training()
    seqtd_obj.run_prediction()

    seqtd_obj.report_perfomance()

def main():
    seqtd_obj = SeqTrajData("../rawdata/1lym.fasta", "../rawdata/rmsf.dat")

    print("===== holdout ========================================================")

    run_data_analysis(seqtd_obj)
    run_data_analysis(seqtd_obj, DecisionTreeRegressor(),'DT' )
    run_data_analysis(seqtd_obj, RandomForestRegressor(), 'RF')

    run_data_analysis(seqtd_obj, MLPRegressor(hidden_layer_sizes=(5, ),
                                    activation='relu',
                                    solver='sgd',
                                    learning_rate='constant',
                                    learning_rate_init=0.001,
                                    max_iter=1000),
                                    'NN')

    print("===== 10-fold crossvalidation ========================================")

    run_data_analysis_w_crossvalidation(seqtd_obj)
    run_data_analysis_w_crossvalidation(seqtd_obj, DecisionTreeRegressor(),'DT' )
    run_data_analysis_w_crossvalidation(seqtd_obj, RandomForestRegressor(), 'RF')

    run_data_analysis_w_crossvalidation(seqtd_obj, MLPRegressor(hidden_layer_sizes=(5, ),
                                    activation='relu',
                                    solver='sgd',
                                    learning_rate='constant',
                                    learning_rate_init=0.001,
                                    max_iter=1000),
                                    'NN')

if __name__ == '__main__':
    main()
