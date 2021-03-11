from SeqTrajData import *                                               

def initialise():
    seqtd_obj = SeqTrajData("../rawdata/1lym.fasta", "../rawdata/rmsf.dat")
    seqtd_obj.create_predictor()
    return(seqtd_obj)

def test_holdout(seqtd):
    seqtd.holdout()
    msg = seqtd.training_x.shape == (114,23)
    return msg

def main():
    seqtd = initialise() 
    test_dict = {
        'holdout' : test_holdout(seqtd)
    }
    for k,v in test_dict.items():
        print("%s : %s" % (k, v))

if __name__ == '__main__':
    main()
