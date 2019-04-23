# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
import numpy as np
import os
import shutil
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.proc.rendering import render_sso_coords_multiprocessing
from syconn.handler.basics import chunkify
from syconn.mp.batchjob_utils import QSUB_script
from syconn.mp.mp_utils import start_multiprocess_obj
from syconn import global_params
path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

ch = args[0]
wd = args[1]
if len(args) > 2:
    render_kwargs = None
else:
    render_kwargs = dict(add_cellobjects=True, woglia=True, overwrite=True)
ssvs_large = []
ssvs_small = []
for ssv_ix in ch:
    sso = SuperSegmentationObject(ssv_ix, working_dir=wd,
                                  enable_locking_so=True)
    if len(sso.sample_locations()) > np.inf:  # TODO: adapt as soon as
        # `render_sso_coords_multiprocessing` has correct sorting of returned views
        ssvs_large.append(sso)
    else:
        ssvs_small.append(sso)

# render huge SSVs in parallel, multiple jobs per SSV
n_parallel_jobs = global_params.NCORES_PER_NODE  # TODO: add as soon as EGL resource allocation
# works
# works # // global_params.NGPUS_PER_NODE
for ssv in ssvs_large:
    render_sso_coords_multiprocessing(ssvs_large, wd, n_parallel_jobs,
                                      render_indexviews=False, return_views=False,
                                      render_kwargs=render_kwargs)
    render_sso_coords_multiprocessing(ssvs_large, wd, n_parallel_jobs,
                                      render_indexviews=True, return_views=False,
                                      render_kwargs=render_kwargs)

# render small SSVs in parallel, one job per SSV
if len(ssvs_small) != 0:
    multi_params = [ssv.id for ssv in ssvs_small]
    multi_params = chunkify(multi_params, n_parallel_jobs)
    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = [(ixs, wd, render_kwargs) for ixs in multi_params]
    path_out = QSUB_script(
        multi_params, "render_views", suffix="_SSV{}".format(ssvs_small[0].id),
        n_cores=1, disable_batchjob=True,
        n_max_co_processes=n_parallel_jobs)
    folder_del = os.path.abspath(path_out + "/../")
    shutil.rmtree(folder_del, ignore_errors=True)
    # TODO: this call leads to an error -> investigate further
    # start_multiprocess_obj('render_views', [[ssv, render_kwargs] for ssv in ssvs_small],
    #                        nb_cpus=n_parallel_jobs)
with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
