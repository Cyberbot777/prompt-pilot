import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def review_and_rewrite_prompt(messy_prompt: str) -> dict:
    """
    Analyze and rewrite a given prompt using the Iris Phase 1 agent.
    Returns a dictionary with review output and rewritten prompt.
    """
    review_template = f"""
Analyze the following prompt for clarity, length, and level of detail.

Check specifically:
1. Is this prompt too long or overly complex?
2. Is this prompt too short or lacking necessary context or details?
3. Are there any ambiguous or unclear phrases?

Prompt:
\"\"\"{messy_prompt}\"\"\"

Provide:
- A clarity rating from 1 to 10.
- Specific issues found.
- Suggestions for improving the prompt.
- A fully rewritten version of the prompt that resolves the identified issues, phrased clearly and professionally.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert prompt engineer."},
            {"role": "user", "content": review_template}
        ]
    )

    return {
        "original_prompt": messy_prompt,
        "review_output": response.choices[0].message.content.strip()
    }
