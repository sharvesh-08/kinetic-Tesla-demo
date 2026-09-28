"""Run the edge-local Layer 2 service."""

import asyncio
import logging

from layer2.config import CONFIG
from layer2.service import Layer2Service

logging.basicConfig(level=logging.INFO, format="%(message)s")
asyncio.run(Layer2Service(CONFIG).run())
