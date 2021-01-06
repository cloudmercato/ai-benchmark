# -*- coding: utf-8 -*-
# Copyright 2019-2020 by Andrey Ignatov. All Rights Reserved.
import logging
from .core import AIBenchmark

handler = logging.StreamHandler()
logger = logging.getLogger('ai_benchmark')
logger.addHandler(handler)

VERSION = (0, 1, 3, 'cm')
__version__ = '.'.join([str(i) for i in VERSION])
