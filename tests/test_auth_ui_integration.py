import unittest
import json
from app import app, pipeline
from utils.auth import create_jwt_token

class TestAuthUIIntegration(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.test_user_a_id = "usr_test_alpha_123"
        self.test_user_a_username = "user_alpha"
        self.test_user_a_email = "alpha@example.com"
        self.test_user_a_password = "password123"

        self.test_user_b_id = "usr_test_beta_456"
        self.test_user_b_username = "user_beta"
        self.test_user_b_email = "beta@example.com"
        self.test_user_b_password = "password456"

        # Clean up existing test users in Mongo or SQLite fallback
        if pipeline and pipeline.cache_manager and pipeline.cache_manager.mongo:
            mongo = pipeline.cache_manager.mongo
            if mongo._available and mongo._db is not None:
                mongo._db.users.delete_many({"email": {"$in": [self.test_user_a_email, self.test_user_b_email]}})
                mongo._db.query_history.delete_many({"user_id": {"$in": [self.test_user_a_id, self.test_user_b_id]}})
            if mongo.fallback_store:
                try:
                    with mongo.fallback_store._get_conn() as conn:
                        conn.execute("DELETE FROM users WHERE email IN (?, ?)", (self.test_user_a_email, self.test_user_b_email))
                        conn.execute("DELETE FROM query_history WHERE user_id IN (?, ?)", (self.test_user_a_id, self.test_user_b_id))
                except Exception:
                    pass

    def test_register_login_and_me_flow(self):
        if not (pipeline and pipeline.cache_manager and pipeline.cache_manager.mongo.available):
            self.skipTest("MongoDB service unavailable")
        # 1. Register User A
        reg_payload = {
            "username": self.test_user_a_username,
            "email": self.test_user_a_email,
            "password": self.test_user_a_password
        }
        res = self.app.post('/api/auth/register', data=json.dumps(reg_payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        token = data.get("token")
        self.assertIsNotNone(token)
        self.assertEqual(data.get("user", {}).get("username"), self.test_user_a_username)

        # 2. Register Duplicate User A should fail
        res_dup = self.app.post('/api/auth/register', data=json.dumps(reg_payload), content_type='application/json')
        self.assertEqual(res_dup.status_code, 400)
        self.assertFalse(res_dup.get_json().get("success"))

        # 3. Login with Email
        login_email_payload = {
            "email": self.test_user_a_email,
            "password": self.test_user_a_password
        }
        res_login_email = self.app.post('/api/auth/login', data=json.dumps(login_email_payload), content_type='application/json')
        self.assertEqual(res_login_email.status_code, 200)
        self.assertTrue(res_login_email.get_json().get("success"))

        # 4. Login with Username
        login_user_payload = {
            "username": self.test_user_a_username,
            "password": self.test_user_a_password
        }
        res_login_user = self.app.post('/api/auth/login', data=json.dumps(login_user_payload), content_type='application/json')
        self.assertEqual(res_login_user.status_code, 200)
        self.assertTrue(res_login_user.get_json().get("success"))

        # 5. Invalid Password Login
        invalid_login = {
            "email": self.test_user_a_email,
            "password": "wrongpassword"
        }
        res_invalid = self.app.post('/api/auth/login', data=json.dumps(invalid_login), content_type='application/json')
        self.assertEqual(res_invalid.status_code, 401)

        # 6. Verify /api/auth/me with valid Bearer token
        headers = {"Authorization": f"Bearer {token}"}
        res_me = self.app.get('/api/auth/me', headers=headers)
        self.assertEqual(res_me.status_code, 200)
        self.assertEqual(res_me.get_json().get("user", {}).get("username"), self.test_user_a_username)

        # 7. Verify /api/auth/me without token -> 401
        res_me_unauth = self.app.get('/api/auth/me')
        self.assertEqual(res_me_unauth.status_code, 401)

    def test_user_isolated_history(self):
        if not (pipeline and pipeline.cache_manager and pipeline.cache_manager.mongo.available):
            self.skipTest("MongoDB not connected")

        # Save history item for User A
        pipeline.cache_manager.mongo.save_query_history(
            user_id=self.test_user_a_id,
            query_hash="hash_alpha_12345",
            original_query="What is Python?",
            normalized_query="what is python",
            intent="broad_search",
            scope="web",
            answer="Python is a programming language.",
            citations=[],
            sources_analyzed=1,
            sources_used=1,
            cache_status="REDIS_HIT"
        )
        h1_id = f"hist_{self.test_user_a_id}_hash_alpha_12345"

        # Save history item for User B
        pipeline.cache_manager.mongo.save_query_history(
            user_id=self.test_user_b_id,
            query_hash="hash_beta_67890",
            original_query="What is MongoDB?",
            normalized_query="what is mongodb",
            intent="broad_search",
            scope="web",
            answer="MongoDB is a NoSQL database.",
            citations=[],
            sources_analyzed=1,
            sources_used=1,
            cache_status="RAG_GENERATED"
        )
        h2_id = f"hist_{self.test_user_b_id}_hash_beta_67890"

        token_a = create_jwt_token(self.test_user_a_id, self.test_user_a_username, self.test_user_a_email)
        token_b = create_jwt_token(self.test_user_b_id, self.test_user_b_username, self.test_user_b_email)

        # User A fetches history
        res_a = self.app.get('/api/history', headers={"Authorization": f"Bearer {token_a}"})
        self.assertEqual(res_a.status_code, 200)
        history_a = res_a.get_json().get("history", [])
        self.assertTrue(any(h.get("_id") == h1_id for h in history_a))
        self.assertFalse(any(h.get("_id") == h2_id for h in history_a))

        # User B fetches history
        res_b = self.app.get('/api/history', headers={"Authorization": f"Bearer {token_b}"})
        self.assertEqual(res_b.status_code, 200)
        history_b = res_b.get_json().get("history", [])
        self.assertTrue(any(h.get("_id") == h2_id for h in history_b))
        self.assertFalse(any(h.get("_id") == h1_id for h in history_b))

        # User A fetches User A item detail
        res_detail_a = self.app.get(f'/api/history/{h1_id}', headers={"Authorization": f"Bearer {token_a}"})
        self.assertEqual(res_detail_a.status_code, 200)
        self.assertEqual(res_detail_a.get_json().get("history_item", {}).get("original_query"), "What is Python?")

        # User B tries to fetch User A item detail -> should return 404 (isolated)
        res_detail_b = self.app.get(f'/api/history/{h1_id}', headers={"Authorization": f"Bearer {token_b}"})
        self.assertEqual(res_detail_b.status_code, 404)

        # User A deletes User A item
        res_del = self.app.delete(f'/api/history/{h1_id}', headers={"Authorization": f"Bearer {token_a}"})
        self.assertEqual(res_del.status_code, 200)
        self.assertTrue(res_del.get_json().get("success"))

if __name__ == '__main__':
    unittest.main()
