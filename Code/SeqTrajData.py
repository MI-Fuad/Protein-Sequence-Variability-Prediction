import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from math import sqrt
from Bio import SeqIO

class SeqTrajData:


    def __init__(self, fasta_filename,rmsf_filename):
        self.fasta_filename = fasta_filename
        self.rmsf_filename = rmsf_filename

        self.input = None
        self.output = None

        self.sequences = self._parse_fasta_file()
        self.rmsf_vector = self._load_rmsf_data()
        self.n_res = len(self.rmsf_vector)

        self.freq_matrix = self._calculate_freq_matrix()
        self.entropy = self._calculate_entropy()
        self.max_entropy = max(self.entropy.iloc[:,0])

        self.final_dataset = self._final_dataset()

        self.training_x = None
        self.training_y = None
        self.test_x = None
        self.test_y = None
        self.predicted_y = None
        self.test_out = None
        self.MSE = None
        self.MAE = None
        self.RMSE = None

        self.k_number_of_folds = None
        self.fold_id = None

        self.predictor = None
        self.predictor_name = None

    def _parse_fasta_file(self):
        sequences = []
        for sequence_record in SeqIO.parse(self.fasta_filename, "fasta"):
            sequences.append(sequence_record.seq)
        return(np.matrix(sequences))

    def _load_rmsf_data(self):
        loadrmsfdata = pd.read_csv(self.rmsf_filename, sep=" ", header=None)
        return(loadrmsfdata)

    def _calculate_freq_matrix(self):
        dataframe = pd.DataFrame(self.sequences)
        counts = dataframe.apply(pd.value_counts)
        frequencymatrix = counts/len(dataframe)
        return(frequencymatrix.fillna(0).T)

    def _calculate_entropy(self):
        entropypercolumn = pd.DataFrame(self.freq_matrix.apply(entropy, axis=1))
        entropypercolumn.rename(columns = {0:"entropy"})
        return entropypercolumn

    def _final_dataset(self):
        self.input = pd.concat([self.freq_matrix, self.rmsf_vector], axis=1)
        self.output = self.entropy
        finaldataset = pd.concat([self.input, self.output], axis=1)
        return finaldataset

    def set_entropy_output(self):
        self.output = self.entropy

    def create_predictor(self, method=LinearRegression(), predictor_name='LR'):
        self.predictor = method
        self.predictor_name = predictor_name

    def holdout(self, test_percentage=30):
        x_training, x_test, y_training, y_test = train_test_split(
            self.input, self.output, test_size=(test_percentage/100), random_state=0)
        self.training_x = x_training
        self.training_y = y_training
        self.test_x = x_test
        self.test_y = y_test
        self.fold_id = 0

    def run_crossvalidation(self, n_splits = 10):
        self.k_number_of_folds = KFold(n_splits)
        self.fold_id = 0

        for train_index,test_index in self.k_number_of_folds.split(self.input):

            self.fold_id += 1

            x_training = self.input.iloc[train_index]
            x_test = self.input.iloc[test_index]

            y_training = self.output.iloc[train_index]
            y_test = self.output.iloc[test_index]

            self.training_x = x_training
            self.training_y = y_training
            self.test_x = x_test
            self.test_y = y_test

            self.run_training()
            self.run_prediction()
            self.report_perfomance()

    def run_training(self):
        self.predictor.fit(self.training_x, np.ravel(self.training_y))

    def run_prediction(self, save_y=True):
        self.predicted_y = pd.DataFrame(self.predictor.predict(self.test_x))
        self.predicted_y.index = self.test_y.index
        abs_diff = abs(self.test_y - self.predicted_y)
        self.test_out = pd.concat([self.test_y, self.predicted_y, abs_diff], axis = 1)
        self.test_out.columns = ['actual','predicted','abs_difference']
        self.MSE = mean_squared_error(self.test_y, self.predicted_y)
        self.MAE = mean_absolute_error(self.test_y, self.predicted_y)
        self.RMSE = sqrt(self.MSE)
        if save_y:
            out_filename = "f%02d_m%s_test_out.csv" % (self.fold_id, self.predictor_name)
            self.test_out.to_csv(out_filename)
            out_pml_filename = "f%02d_m%s_test_out.pml" % (self.fold_id, self.predictor_name)
            out_png_filename = "f%02d_m%s_test_out.png" % (self.fold_id, self.predictor_name)
            fout = open(out_pml_filename, 'w')
            fout.write('@../pdb/2lcb.pml\n')
            for i in range(self.n_res):
                if sum((self.test_out.index == i)):
                    abs_diff = self.test_out[self.test_out.index == i].iloc[0,2]
                    b_value = abs_diff / self.max_entropy
                else:
                    b_value = 0
                b_factor_text = "alter resi %d, b=%-8.3f\n" % (i+1, b_value)
                fout.write(b_factor_text)
            fout.write("cmd.spectrum('b')\n")
            fout.write("ray 800,800 \n")
            fout.write("png %s\n" % out_png_filename)
            fout.close()

    def report_perfomance(self):
        print("Fold: %2d - MSE (%s): %8.3f" % (self.fold_id, self.predictor_name, self.MSE), "MAE (%s): %8.3f" % (self.predictor_name,self.MAE), "RMSE (%s): %8.3f" % (self.predictor_name,self.RMSE))
