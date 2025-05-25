import redis
import pymongo
import json
from datetime import datetime

# Redis and MongoDB Configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "chat_history"
MONGO_COLLECTION = "conversations"

MAX_REDIS_HISTORY = 5  # Redis stores only the last 5 messages

class ConversationManager:
    def __init__(self):
        self.redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.mongo_client = pymongo.MongoClient(MONGO_URI)
        self.db = self.mongo_client[MONGO_DB_NAME]
        self.collection = self.db[MONGO_COLLECTION]

    def add_message(self, role: str, content: str):
        """Stores messages as plain text (no embeddings)."""
        message = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        
        self.redis_client.rpush("chat_history", json.dumps(message))
        if self.redis_client.llen("chat_history") > MAX_REDIS_HISTORY:
            old_message = json.loads(self.redis_client.lpop("chat_history"))
            self.collection.insert_one(old_message)

    def get_relevant_context(self, query: str, top_n=5):
        """Retrieves the most recent chat messages."""
        redis_messages = [json.loads(msg) for msg in self.redis_client.lrange("chat_history", -MAX_REDIS_HISTORY, -1)]
        return [{"role": msg["role"], "content": msg["content"]} for msg in redis_messages[:top_n]]

    def flush_redis_to_mongo(self):
        """Moves all remaining Redis chat history to MongoDB on session end."""
        while self.redis_client.llen("chat_history") > 0:
            old_message = json.loads(self.redis_client.lpop("chat_history"))
            self.collection.insert_one(old_message)
