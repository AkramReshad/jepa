# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import importlib
import logging
import sys

import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dl_src.semantic_mask import main as dl_main
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(app, args, resume_preempt=False):

    logger.info(f'Running pre-training of app: {app}')
    # return importlib.import_module(f'DL-src.semantic_mask').main(
    #     args=args,
    #     resume_preempt=resume_preempt)
    return dl_main()