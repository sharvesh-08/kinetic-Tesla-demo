"""Run the Layer 7 advisory service."""

import asyncio

from .service import Layer7Service

if __name__ == "__main__":
    asyncio.run(Layer7Service().run())
