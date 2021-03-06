# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Jörgen Kornfeld

import sys

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.extraction import cs_extraction_steps
from syconn.mp.mp_utils import start_multiprocess
from syconn import global_params
from syconn.handler.basics import chunkify_successive

if __name__ == '__main__':
    path_storage_file = sys.argv[1]
    path_out_file = sys.argv[2]

    with open(path_storage_file, 'rb') as f:
        args = []
        while True:
            try:
                args.append(pkl.load(f))
            except EOFError:
                break

    if len(args[0]) > 0:
        cs_extraction_steps._write_props_to_syn_singlenode_thread(args)

    with open(path_out_file, "wb") as f:
        pkl.dump(None, f)
