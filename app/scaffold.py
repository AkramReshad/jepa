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

from dl_src.semantic_mask import main as semantic_mask_main
from dl_src.graph import main as graph_main
from dl_src.output_mask import main as output_mask_main

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(app, args, resume_preempt=False):

    logger.info(f'Running pre-training of app: {app}')
    # return importlib.import_module(f'DL-src.semantic_mask').main(
    #     args=args,
    #     resume_preempt=resume_preempt)
    config_to_main = {
        'semantic_mask': semantic_mask_main,
        'graph': graph_main,
        'output_mask': output_mask_main
    }
    print
    if app in config_to_main:
        main_func = config_to_main[app]
    else:
        print("error finding app called {app}")
        return 0
        
    return main_func()