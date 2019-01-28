# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import sys
sys.path
sys.path.append('/u/mariakaw/dev/knossos_utils/')
sys.path.append('/u/mariakaw/dev/SyConn/')

from syconn.exec import exec_syns


if __name__ == '__main__':
    exec_syns.run_syn_analysis()
