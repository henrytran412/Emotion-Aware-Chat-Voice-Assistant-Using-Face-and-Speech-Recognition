"""
User Service for managing user accounts and profiles.
"""

import uuid
from datetime import date, datetime
from typing import Dict, Optional


class UserService:
    def __init__(self):
        self.users: Dict[str, dict] = {}

    def create_user(self, name: str, birthday: date, age: int) -> dict:
        user_id = str(uuid.uuid4())[:8]

        user = {
            "user_id": user_id,
            "name": name,
            "birthday": birthday.isoformat(),
            "age": age,
            "created_at": datetime.now().isoformat(),
        }

        self.users[user_id] = user
        return user

    def get_user(self, user_id: str) -> Optional[dict]:
        return self.users.get(user_id)

    def update_user(self, user_id: str, name: Optional[str] = None) -> Optional[dict]:
        user = self.users.get(user_id)
        if not user:
            return None

        if name:
            user["name"] = name

        return user

    def delete_user(self, user_id: str) -> bool:
        if user_id in self.users:
            del self.users[user_id]
            return True
        return False

    def list_users(self) -> list:
        return list(self.users.values())
