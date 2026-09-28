import unittest

from redis.exceptions import ResponseError

from redis_streams import claim_pending


class Redis5ClaimFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_claim_pending_uses_xpending_and_xclaim(self):
        class Redis5Client:
            def __init__(self):
                self.claimed = None

            async def xautoclaim(self, *_args, **_kwargs):
                raise ResponseError("unknown command 'XAUTOCLAIM'")

            async def xpending_range(self, *_args, **_kwargs):
                return [
                    {"message_id": "1-0", "time_since_delivered": 1500},
                    {"message_id": "2-0", "time_since_delivered": 5},
                ]

            async def xclaim(self, _stream, _group, consumer, _idle, message_ids):
                self.claimed = (consumer, message_ids)
                return [(message_ids[0], {"payload": "recovered"})]

        client = Redis5Client()
        cursor, entries, deleted = await claim_pending(
            client, "source", "workers", "new-consumer", 1000, count=10
        )
        self.assertEqual(cursor, "0-0")
        self.assertEqual(client.claimed, ("new-consumer", ["1-0"]))
        self.assertEqual(entries[0][1]["payload"], "recovered")
        self.assertEqual(deleted, [])


if __name__ == "__main__":
    unittest.main()
