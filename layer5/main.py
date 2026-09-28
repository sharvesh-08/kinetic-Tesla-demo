"""Run the Layer 5 feature-window service."""

import asyncio

from .service import Layer5Service

if __name__ == "__main__":
    asyncio.run(Layer5Service().run())
