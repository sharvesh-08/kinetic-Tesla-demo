"""Layer 4 edge-service entry point."""

import asyncio
import logging

from layer4.service import Layer4Service

logging.basicConfig(level=logging.INFO, format="%(message)s")
asyncio.run(Layer4Service().run())
