import cPickle as pkl
import getpass
from multiprocessing import cpu_count, Process
import multiprocessing.pool
import numpy as np
import os
import re
import shutil
import string
import subprocess
from syconn.utils.basics import negative_to_zero
import sys
import time

__QSUB__ = True
try:
    subprocess.check_output('qstat', shell=True)
except subprocess.CalledProcessError:
    print "QSUB not found, switching to single node multiprocessing."
    __QSUB__ = False

qsub_queue_dict = {"single": "", "half": "", "full": ""}

path_to_scripts = os.path.dirname(__file__)
work_folder = "/home/%s/QSUB/" % getpass.getuser()
username = getpass.getuser()
python_path = sys.executable


def QSUB_script(params, name, queue="single", sge_additional_flags='',
                suffix="", job_name="default", create_random_job_name=True):
    if job_name == "default":
        if create_random_job_name:
            letters = string.ascii_lowercase
            job_name = "".join([letters[l] for l in np.random.randint(0, len(letters), 10)])
            print "Random job_name created: %s" % job_name
        else:
            print "WARNING: running multiple jobs via qsub is only supported with non-default job_names"

    if len(job_name) > 10:
        print "WARNING: Your job_name is longer than 10. job_names have to be distinguishable " \
              "with only using their first 10 characters."

    if os.path.exists(work_folder+"/%s_folder%s/" % (name, suffix)):
        shutil.rmtree(work_folder+"/%s_folder%s/" % (name, suffix))

    path_to_script = path_to_scripts + "/QSUB_%s.py" % (name)
    path_to_storage = work_folder+"/%s_folder%s/storage/" % (name, suffix)
    path_to_sh = work_folder+"/%s_folder%s/sh/" % (name, suffix)
    path_to_log = work_folder+"/%s_folder%s/log/" % (name, suffix)
    path_to_err = work_folder+"/%s_folder%s/err/" % (name, suffix)
    path_to_out = work_folder+"/%s_folder%s/out/" % (name, suffix)

    if queue in qsub_queue_dict:
        sge_queue = qsub_queue_dict[queue]
    else:
        sge_queue = queue

    #TODO: check if queue exists

    if not os.path.exists(path_to_storage):
        os.makedirs(path_to_storage)
    if not os.path.exists(path_to_sh):
        os.makedirs(path_to_sh)
    if not os.path.exists(path_to_log):
        os.makedirs(path_to_log)
    if not os.path.exists(path_to_err):
        os.makedirs(path_to_err)
    if not os.path.exists(path_to_out):
        os.makedirs(path_to_out)

    print "Number of jobs:", len(params)

    time_start = time.time()
    for ii in range(len(params)):
        this_storage_path = path_to_storage+"job_%d.pkl" % ii
        this_sh_path = path_to_sh+"job_%d.sh" % ii
        this_out_path = path_to_out+"job_%d.pkl" % ii
        job_log_path = path_to_log + "job_%d.log" % ii
        job_err_path = path_to_err + "job_%d.log" % ii

        with open(this_sh_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("{0} {1} {2} {3}".format(python_path,
                                             path_to_script,
                                             this_storage_path,
                                             this_out_path))

        with open(this_storage_path, "wb") as f:
            for param in params[ii]:
                pkl.dump(param, f)

        os.chmod(this_sh_path, 0744)

        subprocess.call("qsub -q {0} -o {1} -e {2} -N {3} {4} {5}".format(
            sge_queue,
            job_log_path,
            job_err_path,
            job_name,
            sge_additional_flags,
            this_sh_path), shell=True)

    print "All jobs are submitted: %s" % name
    while True:
        process = subprocess.Popen("qstat -u %s" % username,
                                   shell=True, stdout=subprocess.PIPE)
        nb_lines = 0
        for line in iter(process.stdout.readline, ''):
            if job_name[:10] in line:
                nb_lines += 1
        if nb_lines == 0:
            sys.stdout.write('\rAll jobs were finished in %.2fs\n' % (time.time()-time_start))
            break
        else:
            progress = 100*(len(params) - negative_to_zero(nb_lines))/float(len(params))
            sys.stdout.write('\rProgress: %.2f%% in %.2fs' % (progress, time.time()-time_start))
            sys.stdout.flush()
        time.sleep(1.)

    return path_to_out


def delete_jobs_by_name(job_name, username="sdorkenw"):
    process = subprocess.Popen("qstat -u %s" % username,
                shell=True,
                stdout=subprocess.PIPE)
    job_ids = []
    for line in iter(process.stdout.readline, ''):
        if job_name[:10] in line:
            job_ids.append(re.findall("[\d]+", line)[0])

    command = "qdel "
    for job_id in job_ids:
        command += job_id + ", "
    command = command[:-2]

    process = subprocess.Popen(command,
                shell=True,
                stdout=subprocess.PIPE)


def start_multiprocess(func, params, debug=False, verbose=False, nb_cpus=None):
    """

    Parameters
    ----------
    func : function
    params : function parameters
    debug : bool
    nb_cpus : int

    Returns
    -------
    list of function returns
    """
    # found NoDaemonProcess on stackexchange by Chris Arndt - enables
    # multprocessed grid search with gpu's
    class NoDaemonProcess(Process):
        # make 'daemon' attribute always return False
        def _get_daemon(self):
            return False

        def _set_daemon(self, value):
            pass
        daemon = property(_get_daemon, _set_daemon)

    # We sub-class multi_proc.pool.Pool instead of multi_proc.Pool
    # because the latter is only a wrapper function, not a proper class.
    class MyPool(multiprocessing.pool.Pool):
        Process = NoDaemonProcess

    if nb_cpus is None:
        nb_cpus = max(cpu_count(), 1)

    if debug:
        nb_cpus = 1

    if verbose:
        print "Computing %d parameters with %d cpus." % (len(params), nb_cpus)

    start = time.time()
    if not debug:
        pool = MyPool(nb_cpus)
        result = pool.map(func, params)
        pool.close()
        pool.join()
    else:
        result = map(func, params)

    if verbose:
        print "\nTime to compute grid:", time.time() - start

    return result
