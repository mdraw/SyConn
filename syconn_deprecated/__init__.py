import sip
try:
    sip.setapi('QString', 2)
    sip.setapi('QVariant', 2)
except ValueError:
    sip.setapi('QString', 1)
    sip.setapi('QVariant', 1)
import multi_proc
import utils
import processing
from conmatrix import type_sorted_wiring
from brainqueries import enrich_tracings_all, detect_synapses, \
    predict_celltype_label, type_sorted_wiring
from utils.datahandler import get_filepaths_from_dir