import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        # Ensure you create a .env file in the engine/ directory with your GEMINI_API_KEY
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[WARNING] GEMINI_API_KEY not found in environment. LLM will mock responses.")
            self.mock_mode = True
        else:
            self.mock_mode = False
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest')

        self.system_prompt = """
        You are an energetic, professional live sports commentator. 
        You will receive a JSON payload containing the physical data of an event that just happened. 
        Respond with a single, punchy, exciting sentence of commentary based ONLY on this JSON data. 
        Do not summarize the JSON structure. Just speak the commentary.
        """

    async def generate_commentary(self, json_payload):
        """
        Takes the Enriched JSON from the Delta Filter and asks the LLM to narrate it.
        """
        if self.mock_mode:
            event_type = json_payload.get("event", "UNKNOWN EVENT")
            return f"Whoa, we just saw a massive {event_type} unfold right there on the field!"

        try:
            prompt = f"{self.system_prompt}\n\nLive Event JSON:\n{json_payload}"
            # For streaming tokens, we would use generate_content(stream=True)
            # For the MVP, we generate the short sentence and then stream it char by char to UI
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return "An incredible moment, absolutely stunning!"
