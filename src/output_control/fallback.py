"""
Fallback Handler: Manages LLM output verification failures and retry mechanisms.
When initial processing fails verification, this module handles correction attempts using fallback prompts.
"""

import os
from typing import Optional, List, Tuple

from src.models.factory import AssistantFactory

from src.models.factory import AssistantFactory
from ..tokenizer_utils import decode
from .processor import OutputProcessor
from .verification import VerificationResult
from src.models import get_messages

from configs.config import PROMPT_DIR


class FallbackHandler:
    """
    Handles fallback retry logic when LLM output fails verification.
    
    Strategy:
    1. First attempt: initial LLM output → post-process → verify
    2. If fails (except nesting): use fallback prompt to correct errors
    3. Fallback attempt: error description + failed output → retry → verify
    4. Return best result (original if both fail, corrected if first fails but fallback passes)
    """
    
    def __init__(self, processor: Optional[OutputProcessor] = None):
        """
        Initialize the fallback handler.
        
        Args:
            processor: OutputProcessor instance. If None, creates a new one.
        """
        self.processor = processor or OutputProcessor()
    
    def attempt_correction(
        self,
        assistant: AssistantFactory,
        corrected_output: list,
        original_chunk: list,
        status: VerificationResult,
        fallback_prompt_filename: Optional[str],
        allowed_labels: Optional[List[str]] = None,
    ) -> Tuple[list, bool, dict]:
        """
        Attempt to correct a failed LLM output using fallback mechanism.
        
        Args:
            model: LLM model instance with generate() method
            raw_output: Initial failed LLM output
            original_chunk: Original token list
            verification_result: VerificationResult from initial attempt
            label_config: Label configuration dict
            allowed_labels: Optional list of allowed labels
            fallback_prompt_path: Path to fallback system prompt file
            cot: Whether to use chain-of-thought
        
        Returns:
            Tuple of (corrected_tokens, success: bool, metadata: dict)
            - corrected_tokens: Best available tokens (original, first attempt, or corrected)
            - success: Whether correction passed verification
            - metadata: Dict with details about the correction attempt
                - 'attempt': 'initial' or 'fallback'
                - 'passed': boolean
                - 'error_type': error type if failed
                - 'fallback_used': whether fallback was triggered
        """
        # Load fallback prompt

        with open(PROMPT_DIR / fallback_prompt_filename, 'r', encoding='utf-8') as f:
            fallback_system_prompt = f.read()
            
        
        # Build fallback user prompt
        fallback_user_prompt = self._build_fallback_userprompt(
            original_text=decode(original_chunk),
            failed_output=decode(corrected_output),
            error_type=status.error_type,
            error_details=status.details
        )
        
        # Generate fallback output
        messages = get_messages(system_prompt=fallback_system_prompt, user_input=fallback_user_prompt, fewshot_examples=None, has_system_role=assistant.has_system_role)
        print("  [fallback] about to call assistant.generate")
        fallback_output = assistant.generate(messages=messages)
        print("  [fallback] generate returned")
        
        
        # Process fallback output
        corrected_tokens, correction_result = self.processor.process(
            raw_llm_output=fallback_output,
            original_chunk=original_chunk,
            allowed_labels=allowed_labels,
        )
        
        return corrected_tokens, correction_result.passed, {
            'attempt': 'fallback',
            'passed': correction_result.passed,
            'error_type': correction_result.error_type,
            'error_details': correction_result.details,
            'fallback_used': True
        }
    
    def _build_fallback_userprompt(
        self,
        original_text: str,
        failed_output: str,
        error_type: str,
        error_details: str
    ) -> str:
        """
        Build the user prompt for fallback LLM call.
        
        Args:
            original_text: Original paragraph text
            failed_output: The failed annotated output
            error_type: Type of verification error (hallucination, consistency, etc.)
            error_details: Detailed error message
        
        Returns:
            Formatted user prompt for fallback correction
        """
        prompt = f"""## Input

[ORIGINAL PARAGRAPH]
{original_text}

---

[ANNOTATED PARAGRAPH (Failed)]
{failed_output}

---

[VERIFICATION ERROR]
Type: {error_type}
Details: {error_details}

---

## Task

Correct the annotated paragraph to fix the {error_type} error while preserving the original text content.
Return ONLY the corrected annotated paragraph, nothing else.

## Output

"""
        return prompt
    

    def handle_failure(
        self,
        assistant,
        corrected_output: list,
        original_chunk: list,
        initial_status,
        allowed_labels: Optional[List[str]] = None,
        fallback_prompt_filename: Optional[str] = None,
    ) -> Tuple[list, dict]:
        """
        Handle verification failures with optional fallback.
        
        If error is other types:
        Attempt fallback correction
        
        Args:
            assistant: LLM assistant instance
            corrected_output: Corrected LLM output
            original_chunk: Original tokens
            original_text: Decoded text
            initial_status: Initial VerificationResult
            allowed_labels: Optional allowed labels
            fallback_prompt_filename: Filename for fallback prompt
        
        Returns:
            Tuple of (best_tokens, status_dict)
            - best_tokens: Best available token list
            - status_dict: Dict with status info
                - 'passed': final verification status
                - 'error_type': final error type (or None if passed)
                - 'error_details': final error details
                - 'fallback_used': whether fallback was attempted
                - 'fallback_passed': whether fallback succeeded
        """
        
        
        corrected_tokens, fallback_passed, fallback_meta = self.attempt_correction(
            assistant=assistant,
            corrected_output=corrected_output,
            original_chunk=original_chunk,
            status=initial_status,
            allowed_labels=allowed_labels,
            fallback_prompt_filename=fallback_prompt_filename,
        )

        if fallback_passed:
            return corrected_tokens, {
                'passed': fallback_passed,
                'error_type': fallback_meta.get('error_type') if not fallback_passed else None,
                'error_details': fallback_meta.get('error_details') if not fallback_passed else None,
                'fallback_used': fallback_meta.get('fallback_used', False),
                'fallback_passed': fallback_passed
            }
        
        else:
            # If fallback also fails, return original tokens with failure status unless it was a nesting error

            if initial_status.error_type != 'nesting':
                return original_chunk, {
                    'passed': False,
                    'error_type': fallback_meta.get('error_type'),
                    'error_details': fallback_meta.get('error_details'),
                    'fallback_used': fallback_meta.get('fallback_used', False),
                    'fallback_passed': fallback_passed
                }  
            else:
                return corrected_output, {
                    'passed': False,
                    'error_type': initial_status.error_type,
                    'error_details': initial_status.details,
                    'fallback_used': False,
                    'fallback_passed': False
            }
