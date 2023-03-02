import os, sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
par_dir = os.path.dirname(cur_dir)
csrc_dir = os.path.dirname(par_dir)
lightseq_dir = os.path.dirname(csrc_dir)

sys.path.insert(0, lightseq_dir)
sys.path.insert(0, os.path.dirname(lightseq_dir))
