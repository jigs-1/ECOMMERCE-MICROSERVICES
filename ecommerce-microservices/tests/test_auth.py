import os
import unittest

os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("JWT_ALGORITHM", "HS256")

from shared.auth import create_access_token, decode_access_token, hash_password, verify_password


class AuthTests(unittest.TestCase):
    def test_password_hash_round_trip(self):
        hashed = hash_password("super-secret")
        self.assertTrue(verify_password("super-secret", hashed))
        self.assertFalse(verify_password("wrong-password", hashed))

    def test_token_round_trip(self):
        token = create_access_token("42", {"email": "fanatics@example.com"})
        payload = decode_access_token(token)
        self.assertEqual(payload["sub"], "42")
        self.assertEqual(payload["email"], "fanatics@example.com")


if __name__ == "__main__":
    unittest.main()
