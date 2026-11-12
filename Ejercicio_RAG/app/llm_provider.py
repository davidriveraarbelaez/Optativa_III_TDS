
import os

from google import genai
from groq import Groq


class LLMProvider:

    def __init__(self):
        self.provider = os.getenv(
            "LLM_PROVIDER", "gemini"
        ).lower()

        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise ValueError(
                    "Falta configurar GEMINI_API_KEY"
                )

            self.model = os.getenv("GEMINI_MODEL")

            if not self.model:
                raise ValueError(
                    "Falta configurar GEMINI_MODEL"
                )

            self.client = genai.Client(
                api_key=api_key
            )

        elif self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")

            if not api_key:
                raise ValueError(
                    "Falta configurar GROQ_API_KEY"
                )

            self.model = os.getenv("GROQ_MODEL")

            if not self.model:
                raise ValueError(
                    "Falta configurar GROQ_MODEL"
                )

            self.client = Groq(
                api_key=api_key
            )

        else:
            raise ValueError(
                "Proveedor no soportado. "
                "Utilice gemini o groq."
            )

    def generate(self, prompt):

        if self.provider == "gemini":

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            return response.text

        if self.provider == "groq":

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            return response.choices[0].message.content
