"""
LLM Agent Integration Module
Handles communication with different LLM providers for live commentary generation
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMAgentPool:
    """
    Manages a pool of LLM agents for commentary generation.
    Supports multiple providers: OpenAI, Google Gemini, Anthropic Claude, etc.
    """
    
    def __init__(self):
        self.active_agents = ["GPT-4o-mini"]  # Default agent
        self.agent_configs = self._load_agent_configs()
        self.context_window = {}  # Maintain context for each agent
        self.commentary_history = []
    
    def _load_agent_configs(self) -> Dict:
        """Load agent configurations from environment or default config"""
        return {
            "GPT-4o-mini": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key_env": "OPENAI_API_KEY",
                "max_tokens": 150,
                "temperature": 0.7
            },
            "Gemini 1.5 Flash": {
                "provider": "google",
                "model": "gemini-1.5-flash",
                "api_key_env": "GOOGLE_API_KEY",
                "max_tokens": 150,
                "temperature": 0.7
            },
            "Claude 3.5": {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20241022",
                "api_key_env": "ANTHROPIC_API_KEY",
                "max_tokens": 150,
                "temperature": 0.7
            }
        }
    
    def generate_commentary(self, event_description: str, domain_state: Dict, domain: str = "Cricket") -> str:
        """
        Generate live commentary from the primary active agent.
        Falls back to secondary agents if primary fails.
        
        Args:
            event_description: Description of the event that occurred
            domain_state: Current domain state
            domain: The application domain (Cricket, Security, Traffic)
        
        Returns:
            Generated commentary string
        """
        
        # Build the prompt with context
        system_prompt = self._build_system_prompt(domain_state, domain)
        user_prompt = self._build_user_prompt(event_description, domain_state, domain)
        
        # Try primary agent first
        for agent in self.active_agents:
            try:
                commentary = self._call_agent(agent, system_prompt, user_prompt)
                if commentary:
                    self._save_commentary(agent, event_description, commentary)
                    return commentary
            except Exception as e:
                logger.warning(f"Agent {agent} failed: {e}")
                continue
        
        # Fallback to default commentary if all agents fail
        return self._generate_default_commentary(event_description, domain_state, domain)
    
    def _build_system_prompt(self, domain_state: Dict, domain: str) -> str:
        """Build the system prompt for the LLM with domain-specific context"""
        
        if domain == "Cricket":
            context = f"""- Score: {domain_state.get('runs')} runs, {domain_state.get('wickets')} wickets
- Overs: {domain_state.get('overs')}.{domain_state.get('balls')}
- Striker: {domain_state.get('striker')}
- Bowler: {domain_state.get('bowler')}"""
            persona = "professional cricket commentator providing live, engaging commentary"
            style = "Use cricket-specific language and terminology."
        elif domain == "Security":
            context = f"""- Persons Count: {domain_state.get('person_count')}
- Alert Level: {domain_state.get('security_level')}
- Location: {domain_state.get('location')}"""
            persona = "highly alert security intelligence agent monitoring live feeds"
            style = "Be formal, precise, and emphasize safety and threat levels."
        elif domain == "Traffic":
            context = f"""- Vehicle Count: {domain_state.get('vehicle_count')}
- Incident Count: {domain_state.get('incident_count')}
- Avg Speed: {domain_state.get('avg_speed')} km/h
- Intersection: {domain_state.get('intersection')}"""
            persona = "traffic flow analyst providing real-time infrastructure reports"
            style = "Focus on congestion levels, flow efficiency, and incidents."
        else:
            context = str(domain_state)
            persona = "real-time event analyst"
            style = "Be descriptive and accurate."

        return f"""You are a {persona}.
        
Current {domain} State:
{context}
- Operation Phase: {domain_state.get('phase')}

Your report/commentary should be:
1. Energetic and engaging (if applicable) or precise and professional
2. Technically accurate for the {domain} domain
3. Concise (1-2 sentences)
4. Descriptive of the action and its implications
5. Build on previous events to maintain narrative flow
6. {style}

Generate content that sounds like a live {domain} monitoring feed output."""

    
    def _build_user_prompt(self, event_description: str, domain_state: Dict, domain: str) -> str:
        """Build the user prompt with the current event"""
        
        return f"""Based on the {domain} state above, provide live report/commentary for this event:

EVENT: {event_description}

Current Phase: {domain_state.get('phase')}

Provide 1-2 sentences of engaging live analysis:"""
    
    def _calculate_run_rate(self, match_state: Dict) -> float:
        """Calculate current run rate"""
        overs = match_state.get('overs', 0)
        balls = match_state.get('balls', 0)
        runs = match_state.get('runs', 0)
        
        total_balls = overs * 6 + balls
        if total_balls == 0:
            return 0
        
        return (runs * 6) / total_balls
    
    def _call_agent(self, agent_name: str, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Call the specified agent to generate commentary.
        This is a template method that should be implemented with actual API calls.
        """
        
        config = self.agent_configs.get(agent_name)
        if not config:
            return None
        
        provider = config.get('provider')
        
        try:
            if provider == "openai":
                return self._call_openai(agent_name, config, system_prompt, user_prompt)
            elif provider == "google":
                return self._call_gemini(agent_name, config, system_prompt, user_prompt)
            elif provider == "anthropic":
                return self._call_claude(agent_name, config, system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"Error calling {agent_name}: {e}")
            return None
    
    def _call_openai(self, agent_name: str, config: Dict, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call OpenAI API"""
        try:
            import openai
            
            api_key = os.getenv(config['api_key_env'])
            if not api_key:
                logger.warning(f"API key not found for {agent_name}")
                return None
            
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=config['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=config['max_tokens'],
                temperature=config['temperature']
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None
    
    def _call_gemini(self, agent_name: str, config: Dict, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call Google Gemini API"""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv(config['api_key_env'])
            if not api_key:
                logger.warning(f"API key not found for {agent_name}")
                return None
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=config['model'],
                system_instruction=system_prompt
            )
            
            response = model.generate_content(
                user_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=config['max_tokens'],
                    temperature=config['temperature']
                )
            )
            
            return response.text
        
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None
    
    def _call_claude(self, agent_name: str, config: Dict, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call Anthropic Claude API"""
        try:
            import anthropic
            
            api_key = os.getenv(config['api_key_env'])
            if not api_key:
                logger.warning(f"API key not found for {agent_name}")
                return None
            
            client = anthropic.Anthropic(api_key=api_key)
            
            response = client.messages.create(
                model=config['model'],
                max_tokens=config['max_tokens'],
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return response.content[0].text
        
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return None
    
    def _generate_default_commentary(self, event_description: str, domain_state: Dict, domain: str) -> str:
        """Generate default commentary if LLM calls fail"""
        
        if domain == "Cricket":
            templates = {
                "SHOT_PLAYED": "And the batsman plays a shot! {striker} connects well with the ball.",
                "BALL_PITCHED": "The ball has pitched on good length. {bowler} gets good pace and line.",
                "CATCH": "And that's caught! Brilliant fielding gets {striker} out.",
                "WICKET": "And that's the wicket! {bowler} strikes."
            }
        elif domain == "Security":
            templates = {
                "INTRUSION": "Security Alert! Unauthorized entry detected at {location}.",
                "LOITERING": "Suspicious behavior observed near the perimeter.",
                "NORMAL": "Area remains secure under current surveillance."
            }
        elif domain == "Traffic":
            templates = {
                "CONGESTION": "Traffic buildup detected at {intersection}. Flow rate decreasing.",
                "ACCIDENT": "Incident detected! Emergency services may be required at {intersection}.",
                "SPEEDING": "High-speed vehicle detected. Automated log recorded."
            }
        else:
            templates = {}
        
        # Extract event type from description
        event_type = "DEFAULT"
        for key in templates:
            if key.lower() in event_description.lower():
                event_type = key
                break
        
        template = templates.get(event_type, f"Significant activity detected in {domain} feed.")
        
        return template.format(
            striker=domain_state.get('striker', 'Batsman'),
            bowler=domain_state.get('bowler', 'Bowler'),
            location=domain_state.get('location', 'Site'),
            intersection=domain_state.get('intersection', 'Junction')
        )
    
    def _save_commentary(self, agent_name: str, event: str, commentary: str):
        """Save commentary to history for logging/review"""
        
        self.commentary_history.append({
            'timestamp': datetime.now().isoformat(),
            'agent': agent_name,
            'event': event,
            'commentary': commentary
        })
    
    def get_agent_status(self) -> Dict:
        """Get status of all agents"""
        
        status = {}
        for agent in self.agent_configs.keys():
            config = self.agent_configs[agent]
            api_key = os.getenv(config['api_key_env'])
            status[agent] = {
                'configured': bool(api_key),
                'active': agent in self.active_agents,
                'provider': config.get('provider')
            }
        
        return status
    
    def add_context_to_agent(self, agent_name: str, context: Dict):
        """Add custom context to a specific agent"""
        
        if agent_name not in self.context_window:
            self.context_window[agent_name] = []
        
        self.context_window[agent_name].append(context)
    
    def get_commentary_history(self, agent_name: Optional[str] = None) -> List[Dict]:
        """Get commentary history, optionally filtered by agent"""
        
        if agent_name:
            return [c for c in self.commentary_history if c.get('agent') == agent_name]
        
        return self.commentary_history
    
    def clear_history(self):
        """Clear commentary history"""
        self.commentary_history = []


class EventProcessor:
    """Process DSG events and prepare them for LLM consumption"""
    
    @staticmethod
    def process_event(dsg_payload: Dict) -> str:
        """
        Convert DSG payload into natural language event description
        
        Args:
            dsg_payload: JSON payload from DSG engine
        
        Returns:
            Natural language description of the event
        """
        
        event_type = dsg_payload.get('event', 'Unknown Event')
        primary = dsg_payload.get('primary_actor', {})
        secondary = dsg_payload.get('secondary_actor', {})
        
        descriptions = {
            'SHOT_PLAYED': f"The {primary.get('type')} has made contact with the {secondary.get('type')}. Velocity: {primary.get('velocity_kph')} kph.",
            'BALL_PITCHED': f"The ball has been delivered and pitched. Ball velocity: {primary.get('velocity_kph')} kph.",
            'BALL_CAUGHT': f"The ball has been caught by the fielder.",
            'RUN_SCORED': f"A run has been completed between the wickets.",
            'WICKET_FALLEN': f"The batsman is out! The bowler gets a wicket."
        }
        
        return descriptions.get(event_type, f"Event: {event_type} - {primary.get('type')} interacts with {secondary.get('type')}")


# Example usage
if __name__ == "__main__":
    # Initialize the agent pool
    agent_pool = LLMAgentPool()
    
    # Simulate match state
    match_state = {
        'runs': 45,
        'wickets': 2,
        'overs': 8,
        'balls': 3,
        'striker': 'Virat Kohli',
        'bowler': 'Jasprit Bumrah',
        'phase': 'Middle Overs'
    }
    
    # Test commentary generation
    event = "The bowler is running up from over the wicket. The batsman takes guard outside off stump."
    
    print("Generating commentary...")
    commentary = agent_pool.generate_commentary(event, match_state)
    print(f"Commentary: {commentary}")
    
    # Get agent status
    print("\nAgent Status:")
    for agent, status in agent_pool.get_agent_status().items():
        print(f"  {agent}: {status}")
