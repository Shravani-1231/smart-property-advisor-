"""
Authentication module for Smart Property Advisor
Handles Google OAuth and user account management
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import hashlib
import secrets

# File to store user accounts
USERS_FILE = "data/users.json"


def load_users() -> Dict[str, Any]:
    """Load users from file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_users(users: Dict[str, Any]) -> None:
    """Save users to file"""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)


def hash_password(password: str) -> str:
    """Hash a password"""
    salt = secrets.token_hex(16)
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return salt + pwdhash.hex()


def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verify a password against its hash"""
    salt = stored_password[:32]
    stored_hash = stored_password[32:]
    pwdhash = hashlib.pbkdf2_hmac('sha256', provided_password.encode(), salt.encode(), 100000)
    return pwdhash.hex() == stored_hash


def create_user(email: str, password: str, name: str, auth_type: str = "email") -> bool:
    """Create a new user account"""
    users = load_users()
    
    if email in users:
        return False  # User already exists
    
    users[email] = {
        "name": name,
        "email": email,
        "password": hash_password(password) if password else None,
        "auth_type": auth_type,  # "email" or "google"
        "created_at": datetime.now().isoformat(),
        "last_login": None,
        "is_active": True
    }
    
    save_users(users)
    return True


def verify_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify user credentials"""
    users = load_users()
    
    if email not in users:
        return None
    
    user = users[email]
    
    if not user.get("is_active", True):
        return None
    
    # For Google auth, password is None
    if user["auth_type"] == "google":
        user["last_login"] = datetime.now().isoformat()
        save_users(users)
        return user
    
    # For email auth, verify password
    if user["password"] and verify_password(user["password"], password):
        user["last_login"] = datetime.now().isoformat()
        save_users(users)
        return user
    
    return None


def get_or_create_google_user(email: str, name: str) -> Dict[str, Any]:
    """Get existing Google user or create new one"""
    users = load_users()
    
    if email in users:
        # Existing user - update last login
        users[email]["last_login"] = datetime.now().isoformat()
        save_users(users)
        return users[email]
    else:
        # Create new Google user
        create_user(email, None, name, auth_type="google")
        return load_users()[email]


def user_exists(email: str) -> bool:
    """Check if user exists"""
    users = load_users()
    return email in users


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email"""
    users = load_users()
    return users.get(email)


# Demo users for testing
def init_demo_users():
    """Initialize demo users if none exist"""
    users = load_users()
    
    if not users:
        # Create demo users
        create_user("admin@example.com", "admin123", "Admin User", "email")
        create_user("user@example.com", "user123", "Regular User", "email")
        create_user("demo@example.com", "demo123", "Demo User", "email")
        print("Demo users created!")


if __name__ == "__main__":
    init_demo_users()
    print("Users:", load_users())