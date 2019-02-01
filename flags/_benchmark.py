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
"""Flags for benchmarking models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from ._conventions import help_wrap


def define_benchmark(benchmark_log_dir=True):
    """Register benchmarking flags.
    Args:
        benchmark_log_dir: Create a flag to specify location for benchmark logging.
    Returns:
        A list of flags for core.py to marks as key flags.
    """

    key_flags = []

    flags.DEFINE_enum(
        name="benchmark_logger_type", default="BenchmarkBaseLogger",
        enum_values=["BaseBenchmarkLogger", "BenchmarkFileLogger",
                     "BenchmarkBigQueryLogger"],
        help=help_wrap("The type of benchmark logger to use. Defaults to using "
                       "BaseBenchmarkLogger which logs to STDOUT. Different "
                       "loggers will require other flags to be able to work."))
    flags.DEFINE_string(
        name="benchmark_test_id", short_name="bti", default=None,
        help=help_wrap("The unique test ID of the benchmark run. It could be the "
                       "combination of key parameters. It is hardware "
                       "independent and could be used compare the performance "
                       "between different test runs. This flag is designed for "
                       "human consumption, and does not have any impact within "
                       "the system."))

    if benchmark_log_dir:
        flags.DEFINE_string(
            name="benchmark_log_dir", short_name="bld", default=None,
            help=help_wrap("The location of the benchmark logging.")
        )

    @flags.multi_flags_validator(
        ["benchmark_logger_type", "benchmark_log_dir"],
        message="--benchmark_logger_type=BenchmarkFileLogger will require "
                "--benchmark_log_dir being set")
    def _check_benchmark_log_dir(flags_dict):
        benchmark_logger_type = flags_dict["benchmark_logger_type"]
        if benchmark_logger_type == "BenchmarkFileLogger":
            return flags_dict["benchmark_log_dir"]
        return True

    return key_flags
