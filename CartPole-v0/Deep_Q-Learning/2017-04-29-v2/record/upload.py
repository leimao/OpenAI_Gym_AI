import gym

API_KEY = 'sk_CsKvFpZRwiFZuLsn0225A' # API key for submission
RECORD_FILENAME = 'CartPole-v0-experiment'

gym.upload(RECORD_FILENAME, api_key = API_KEY)
