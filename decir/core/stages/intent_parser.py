"""Stage 1: Intent Parser for DeCIR.

This module implements the first stage of the DeCIR pipeline, which analyzes
the reference image and modification text to extract structured edit intent.
"""

import json
import re
from typing import Dict, Union, Optional
from pathlib import Path
from PIL import Image

from decir.models.qwen_client import Qwen3VLClient
from decir.utils.intent_schema import INTENT_SCHEMA, validate_intent


# Default system prompt for intent parsing
DEFAULT_SYSTEM_PROMPT = """You are an expert vision-language model specialized in analyzing image editing intent.

Given a reference image and a modification text, your task is to:
1. Understand what the user wants to change
2. Identify which objects/regions need to be modified
3. Classify the type of edit operation

Output your analysis as a JSON object following this schema:
{
  "edit_type": "subject_only" | "background_only" | "subject_and_background" | "composition_change" | "replacement" | "global",
  "operations": [
    {
      "action": "add" | "remove" | "replace" | "modify" | "duplicate" | "change_count" | "swap",
      "target": "subject" | "background" | "specific_object" | "global",
      "object": "name of object",
      "attributes": {"key": "value"}
    }
  ],
  "needs_bbox": true | false,
  "needs_background_mask": true | false,
  "needs_global_redraw": true | false
}
"""


class IntentParser:
    """Stage 1: Intent Parser.

    Analyzes user's modification intent from reference image and text description
    using a multi-modal language model (Qwen3-VL).

    The parser extracts structured information including:
    - Edit type (subject-only, background, global, etc.)
    - Specific operations (add, remove, modify, etc.)
    - Target objects and their attributes
    - Requirements for downstream stages (bbox, masks, etc.)

    Attributes:
        qwen_client: Qwen3-VL client for vision-language understanding.
        system_prompt: System prompt defining the parsing task.

    Example:
        >>> from decir.models.qwen_client import Qwen3VLClient
        >>> client = Qwen3VLClient(model_path="Qwen/Qwen2-VL-7B-Instruct")
        >>> parser = IntentParser(qwen_client=client)
        >>> intent = parser.parse(
        ...     reference_image="car.jpg",
        ...     modification_text="change the car color to blue"
        ... )
        >>> print(intent["edit_type"])
        "subject_only"
        >>> print(intent["operations"][0]["action"])
        "modify"
    """

    def __init__(
        self,
        qwen_client: Qwen3VLClient,
        system_prompt: Optional[str] = None,
        validate_schema: bool = True
    ):
        """Initialize Intent Parser.

        Args:
            qwen_client: Initialized Qwen3-VL client.
            system_prompt: Custom system prompt (uses default if None).
            validate_schema: Whether to validate output against JSON schema.
        """
        self.client = qwen_client
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.validate_schema = validate_schema

    def parse(
        self,
        reference_image: Union[str, Path, Image.Image],
        modification_text: str,
        prompt_template: Optional[str] = None
    ) -> Dict:
        """Parse edit intent from reference image and modification text.

        Args:
            reference_image: Reference image (path or PIL Image).
            modification_text: Text description of desired modification.
            prompt_template: Optional custom user prompt template.
                            Use "<<<CAPTION>>>" placeholder for modification text.

        Returns:
            Dictionary containing structured edit intent.

        Raises:
            ValueError: If JSON parsing fails or schema validation fails.
            RuntimeError: If Qwen3-VL generation fails.

        Example:
            >>> intent = parser.parse(
            ...     reference_image="photo.jpg",
            ...     modification_text="add a hat to the person"
            ... )
            >>> print(intent)
            {
                "edit_type": "subject_only",
                "operations": [{
                    "action": "add",
                    "target": "subject",
                    "object": "person",
                    "attributes": {"clothing_item": "hat"}
                }],
                "needs_bbox": True,
                "needs_background_mask": False,
                "needs_global_redraw": False
            }
        """
        # Prepare user prompt
        if prompt_template:
            user_text = prompt_template.replace("<<<CAPTION>>>", modification_text)
        else:
            user_text = f"Reference image modification: {modification_text}"

        # Generate intent using Qwen3-VL
        try:
            output_text = self.client.generate(
                system_prompt=self.system_prompt,
                user_text=user_text,
                image=reference_image
            )
        except Exception as e:
            raise RuntimeError(f"Qwen3-VL generation failed: {e}")

        # Extract JSON from output
        intent = self._extract_json(output_text)

        # Validate against schema
        if self.validate_schema and not validate_intent(intent):
            # Try fallback parsing
            print("[WARNING] Intent validation failed. Attempting fallback...")
            intent = self._fallback_parse(modification_text)

        return intent

    def parse_with_custom_prompt(
        self,
        reference_image: Union[str, Path, Image.Image],
        modification_text: str,
        prompt_file: Union[str, Path]
    ) -> Dict:
        """Parse intent using custom prompt from file.

        This method supports prompt files with separate SYSTEM and USER sections
        marked by "### SYSTEM" and "### USER" headers.

        Args:
            reference_image: Reference image (path or PIL Image).
            modification_text: Text description of modification.
            prompt_file: Path to prompt file.

        Returns:
            Dictionary containing structured edit intent.

        Example prompt file format:
            ### SYSTEM
            You are a vision assistant...

            ### USER
            Analyze the reference image and modification: <<<CAPTION>>>
        """
        # Load prompt file
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_content = f.read()

        # Parse SYSTEM and USER sections
        if "### USER" in prompt_content:
            sys_prompt = (
                prompt_content.split("### USER")[0]
                .replace("### SYSTEM", "")
                .strip()
            )
            user_template = prompt_content.split("### USER")[1].strip()
        else:
            sys_prompt = prompt_content.strip()
            user_template = "Modification: <<<CAPTION>>>"

        # Replace CAPTION placeholder
        user_text = user_template.replace("<<<CAPTION>>>", modification_text)

        # Generate
        output_text = self.client.generate(
            system_prompt=sys_prompt,
            user_text=user_text,
            image=reference_image
        )

        # Extract and validate
        intent = self._extract_json(output_text)

        if self.validate_schema and not validate_intent(intent):
            intent = self._fallback_parse(modification_text)

        return intent

    @staticmethod
    def _extract_json(text: str) -> Dict:
        """Extract JSON object from text response.

        Args:
            text: Text containing JSON (possibly with surrounding text).

        Returns:
            Parsed JSON dictionary.

        Raises:
            ValueError: If no valid JSON is found.
        """
        # Try to find JSON object in curly braces
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError(f"No valid JSON found in response:\n{text}")

        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nText: {match.group()}")

    @staticmethod
    def _fallback_parse(modification_text: str) -> Dict:
        """Fallback intent parser using heuristics.

        This method provides a basic fallback when Qwen3-VL fails to generate
        valid JSON. It uses simple keyword matching.

        Args:
            modification_text: Modification text to parse.

        Returns:
            Basic intent dictionary.
        """
        text_lower = modification_text.lower()

        # Detect action
        if any(kw in text_lower for kw in ["add", "insert", "include"]):
            action = "add"
        elif any(kw in text_lower for kw in ["remove", "delete", "erase"]):
            action = "remove"
        elif any(kw in text_lower for kw in ["replace", "swap", "change to"]):
            action = "replace"
        else:
            action = "modify"

        # Detect target
        if any(kw in text_lower for kw in ["background", "scene", "setting"]):
            edit_type = "background_only"
            target = "background"
        else:
            edit_type = "subject_only"
            target = "subject"

        return {
            "edit_type": edit_type,
            "operations": [{
                "action": action,
                "target": target,
                "object": "unknown",
                "attributes": {"description": modification_text}
            }],
            "needs_bbox": True,
            "needs_background_mask": False,
            "needs_global_redraw": False
        }
