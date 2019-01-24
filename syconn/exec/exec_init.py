# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from knossos_utils import knossosdataset
from knossos_utils import chunky
from syconn import global_params
from syconn.extraction import object_extraction_wrapper as oew
from syconn.proc import sd_proc
from syconn.reps.segmentation import SegmentationDataset
from syconn.handler.logger import initialize_logging


# TODO: make it work with new SyConn
def run_create_sds(chunk_size=None):
    if chunk_size is None:
        chunk_size = [512, 512, 512]
    log = initialize_logging('create_sds', global_params.paths.working_dir + '/logs/',
                             overwrite=False)
    log.info('Generating SegmentationDatasets for cell and cell organelle supervoxels.')
    # TODO: get rid of explicit voxel extraction, all info necessary should be extracted at the beginning, e.g. size, bounding box etc and then refactor to only use those cached attributes!
    # resulting ChunkDataset, required for SV extraction --

    # Sets initial values of object
    kd = knossosdataset.KnossosDataset()
    # Initializes the dataset by parsing the knossos.conf in path + "mag1"
    kd.initialize_from_knossos_path(global_params.paths.kd_seg_path)

    # Object extraction - 2h, the same has to be done for all cell organelles
    cd_dir = global_params.paths.working_dir + "chunkdatasets/sv/"
    # Class that contains a dict of chunks (with coordinates) after initializing it
    cd = chunky.ChunkDataset()
    cd.initialize(kd, kd.boundary, chunk_size, cd_dir,
                  box_coords=[0, 0, 0], fit_box_size=True)
    oew.from_ids_to_objects(cd, "sv", overlaydataset_path=global_params.paths.kd_seg_path, n_chunk_jobs=5000,
                            hdf5names=["sv"], n_max_co_processes=None, qsub_pe='default',
                            qsub_queue='all.q', qsub_slots=1, n_folders_fs=10000)
    log.info('Finished object extraction for cell SVs.')

    log.info('Generating SegmentationDatasets for cell and cell organelle supervoxels.')
    # create SegmentationDataset for each cell organelle
    for co in global_params.existing_cell_organelles:
        cd_dir = global_params.paths.working_dir + "chunkdatasets/{}/".format(co)
        # Class that contains a dict of chunks (with coordinates) after initializing it
        cd = chunky.ChunkDataset()
        cd.initialize(kd, kd.boundary, chunk_size, cd_dir,
                      box_coords=[0, 0, 0], fit_box_size=True)
        prob_kd_path_dict = {co: getattr(global_params.paths, 'kd_{}_path'.format(co))}
        # This creates a SegmentationDataset of type 'co'
        prob_thresh = global_params.paths.entries["Probathresholds"][co]  # get probability threshold
        oew.from_probabilities_to_objects(cd, co, membrane_kd_path=global_params.paths.kd_seg_path,
                                          prob_kd_path_dict=prob_kd_path_dict, thresholds=[prob_thresh],
                                          workfolder=global_params.paths.working_dir,
                                          hdf5names=[co], n_max_co_processes=None, qsub_pe='default',
                                          qsub_queue='all.q', n_folders_fs=10000, debug=False)
        sd_co = SegmentationDataset(obj_type=co, working_dir=global_params.paths.working_dir)
        sd_proc.dataset_analysis(sd_co, qsub_pe="default", qsub_queue='all.q',
                                 stride=10, compute_meshprops=True)
        # About 0.2 h per object class -- TODO: Seems to be slow for VC
        sd_proc.map_objects_to_sv(sd, co, global_params.paths.kd_seg_path, qsub_pe='default',
                                  qsub_queue='all.q', stride=20)
        log.info('Finished object extraction for {} SVs.'.format(co))

    # Object Processing - 0.5h -- Perform after mapping to also cache mapping ratios
    sd = SegmentationDataset("sv", working_dir=global_params.paths.working_dir)
    sd_proc.dataset_analysis(sd, qsub_pe="default", qsub_queue='all.q',
                             stride=10, compute_meshprops=True)
