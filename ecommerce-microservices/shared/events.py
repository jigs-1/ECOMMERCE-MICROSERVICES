import asyncio
import json
from collections.abc import Awaitable, Callable

from redis.asyncio import Redis

from shared.config import get_settings


def get_redis_client() -> Redis:
    settings = get_settings()
    return Redis.from_url(settings.redis_url, decode_responses=True)


async def publish_event(channel: str, payload: dict) -> None:
    redis = get_redis_client()
    try:
        await redis.publish(channel, json.dumps(payload))
    finally:
        await redis.aclose()


async def subscribe_forever(channel: str, handler: Callable[[dict], Awaitable[None]]) -> None:
    redis = get_redis_client()
    pubsub = redis.pubsub()
    await pubsub.subscribe(channel)
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message.get("data"):
                await handler(json.loads(message["data"]))
            await asyncio.sleep(0.1)
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
        await redis.aclose()
