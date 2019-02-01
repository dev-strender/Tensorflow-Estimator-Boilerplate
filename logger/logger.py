# Original source code is from
# https://github.com/tensorflow/models/blob/master/official/utils/logs/logger.py

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Logging Utilities
- logging TF_version, CPU, GPU, MEM info

For collecting local environment metrics like CPU and memory, certain python
packages need to be installed.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import datetime
import os
import contextlib
import multiprocessing
import threading

from absl import flags
from tensorflow.python.client import device_lib

ENV_LOG_FILE_NAME = "runtime_info.log"
_DATE_TIME_FORMAT_PATTERN = "%Y-%m-%dT%H:%M:%S.%fZ"
RUN_STATUS_SUCCESS = "success"
RUN_STATUS_FAILURE = "failure"
RUN_STATUS_RUNNING = "running"

FLAGS = flags.FLAGS

# Don't use it directly. Use get_benchmark_logger to access a logger.
_benchmark_logger = None
_logger_lock = threading.Lock()


def config_benchmark_logger(flag_obj=None):
    """Config the global benchmark logger."""
    _logger_lock.acquire()
    try:
        global _benchmark_logger
        if not flag_obj:
            flag_obj = FLAGS

        if not hasattr(flag_obj, "benchmark_logger_type") or flag_obj.benchmark_logger_type == "BenchmarkBaseLogger":
            _benchmark_logger = BenchmarkBaseLogger()
        # elif flag_obj.benchmark_logger_type == "BenchmarkFileLogger":
        #     _benchmark_logger = BenchmarkFileLogger(flag_obj.benchmark_log_dir)
        else:
            raise ValueError("Unrecognized benchmark_logger_type: %s" % flag_obj.benchmark_logger_type)

    finally:
        _logger_lock.release()
    return _benchmark_logger


def get_benchmark_logger():
    if not _benchmark_logger:
        config_benchmark_logger()
    return _benchmark_logger


@contextlib.contextmanager
def benchmark_context(flag_obj):
    """Context of benchmark, which will update status of the run accordingly."""
    benchmark_logger = config_benchmark_logger(flag_obj)
    try:
        yield
        benchmark_logger.on_finish(RUN_STATUS_SUCCESS)
    except Exception:
        benchmark_logger.on_finish(RUN_STATUS_FAILURE)
        raise


class BenchmarkBaseLogger(object):
    """Class to log the benchmark information to STDOUT."""

    def log_run_info(self, model_name, dataset_name, run_params):
        tf.logging.info("Benchmark run: %s",
                        _gather_run_info(model_name, dataset_name, run_params))

    def on_finish(self, status):
        pass


def _gather_run_info(model_name, dataset_name, run_params):
    """Collect the running information for the local environment"""
    run_info = {
        "model_name": model_name,
        "dataset": dataset_name,
        "machine_config": {},
        "run_date": datetime.datetime.now().strftime(_DATE_TIME_FORMAT_PATTERN)
    }
    session_config = None

    if "session_config" in run_params:
        session_config = run_params["session_config"]
    _collect_tensorflow_info()
    _collect_tensorflow_environment_variables(run_info)
    _collect_run_params(run_info, run_params)
    _collect_cpu_info(run_info)
    _collect_gpu_info(run_info, session_config)
    _collect_memory_info(run_info)

    return run_info


def _collect_tensorflow_info(run_info):
    run_info["tensorflow_version"] = {"version": tf.VERSION, "git_hash": tf.GIT_VERSION}


def _collect_run_params(run_info, run_params):
    """Log the parameter information for the benchmark run."""
    def process_param(name, value):
        type_check = {
            str: {"name": name, "string_value": value},
            int: {"name": name, "long_value": value},
            bool: {"name": name, "bool_value": str(value)},
            float: {"name": name, "float_value": value},
        }
        return type_check.get(type(value),
                              {"name": name, "string_value": str(value)})
    if run_params:
        run_info["run_parameters"] = [
            process_param(k, v) for k, v in sorted(run_params.items())]


def _collect_tensorflow_environment_variables(run_info):
    run_info["tensorflow_environment_variables"] = [
        {"name": k, "value": v}
        for k, v in sorted(os.environ.items()) if k.startswith("TF_")]


def _collect_cpu_info(run_info):
    """Collect the CPU information for the local environment."""
    cpu_info = {}
    cpu_info["num_cores"] = multiprocessing.cpu_count()

    try:
        # Note: cpuinfo is not installed in the TensorFlow OSS tree.
        # It is installable via pip.
        import cpuinfo    # pylint: disable=g-import-not-at-top

        info = cpuinfo.get_cpu_info()
        cpu_info["cpu_info"] = info["brand"]
        cpu_info["mhz_per_cpu"] = info["hz_advertised_raw"][0] / 1.0e6

        run_info["machine_config"]["cpu_info"] = cpu_info
    except ImportError:
        tf.logging.warn("'cpuinfo' not imported. CPU info will not be logged.")


def _collect_gpu_info(run_info, session_config=None):
    """Collect local GPU information by TF device library."""
    gpu_info = {}
    local_device_protos = device_lib.list_local_devices(session_config)

    gpu_info["count"] = len([d for d in local_device_protos
                           if d.device_type == "GPU"])
    # The device description usually is a JSON string, which contains the GPU
    # model info, eg:
    # "device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:00:04.0"
    for d in local_device_protos:
        if d.device_type == "GPU":
            gpu_info["model"] = _parse_gpu_model(d.physical_device_desc)
        # Assume all the GPU connected are same model
        break
    run_info["machine_config"]["gpu_info"] = gpu_info


def _collect_memory_info(run_info):
    try:
        # Note: psutil is not installed in the TensorFlow OSS tree.
        # It is installable via pip.
        import psutil   # pylint: disable=g-import-not-at-top
        vmem = psutil.virtual_memory()
        run_info["machine_config"]["memory_total"] = vmem.total
        run_info["machine_config"]["memory_available"] = vmem.available
    except ImportError:
        tf.logging.warn("'psutil' not imported. Memory info will not be logged.")


def _parse_gpu_model(physical_device_desc):
    # Assume all the GPU connected are same model
    for kv in physical_device_desc.split(","):
        k, _, v = kv.partition(":")
        if k.strip() == "name":
            return v.strip()
    return None


def _convert_to_json_dict(input_dict):
    if input_dict:
        return [{"name": k, "value": v} for k, v in sorted(input_dict.items())]
    else:
        return []

