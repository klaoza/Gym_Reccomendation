import requests
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_as_reranker")

class LLMReranker:
    """
    Extremely fast and stable LLM-based reranker using:
    - llama3.2:3b
    - /api/chat endpoint (much faster and more stable than /generate)
    - Ultra-minimal prompt (critical for small local models)
    - Hard fallback if model fails
    """

    def __init__(self, model="llama3.2:3b",
                 server_url="http://127.0.0.1:11434/api/chat"):
        self.model = model
        self.server_url = server_url

    # ---------------------------------------------------------
    # BUILD MINIMAL PROMPT (critical for speed!)
    # ---------------------------------------------------------
    def _build_prompt(self, user, gym):
       
        return (
            f"Score 0-1: wants {user.get('desired_facilities')}, "
            f"gym has {gym.get('facilities')}."
        )

    # ---------------------------------------------------------
    # PARSE SCORE
    # ---------------------------------------------------------
    def _extract_score(self, text: str) -> float:
        """
        Extract a float between 0 and 1 from LLM output.
        """
        match = re.search(r"0\.\d+", text)
        if match:
            return float(match.group(0))
        # Try detecting edge cases: "1", "0"
        if text.strip() == "1":
            return 1.0
        if text.strip() == "0":
            return 0.0
        return 0.5  # fallback neutral value

    # ---------------------------------------------------------
    # MAIN LLM SCORING METHOD
    # ---------------------------------------------------------
    def score(self, user_profile, gym_profile) -> float:
        prompt = self._build_prompt(user_profile, gym_profile)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return ONLY a number 0 to 1."},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        try:
            response = requests.post(
                self.server_url,
                json=payload,
                timeout=90  
            )
            response.raise_for_status()

            text = response.json()["message"]["content"].strip()
            score = self._extract_score(text)
            return score

        except Exception as e:
            logger.error(f"LLM reranker failed: {e}")
            return 0.5 
