from SeqTrajData import *                                               

def initialise():
    seqtd_obj = SeqTrajData("../rawdata/1lym.fasta", "../rawdata/rmsf.dat")
    return(seqtd_obj)

def test_input(seqtd):
    msg = seqtd.input.shape == (164,23)
    return msg

def main():
    seqtd = initialise() 
    test_dict = {
        'input' : test_input(seqtd)
    }
    for k,v in test_dict.items():
        print("%s : %s" % (k, v))

if __name__ == '__main__':
    main()
