import os
import re
import asyncio
from typing import List, Dict, Any, AsyncIterator
from mistralai import Mistral
import discord

from prompts import DECOMPOSITION_PROMPT, SYNTHESIS_PROMPT

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = "You are a helpful assistant."


class MistralProvider:
    """Mistral-specific implementation of LLM provider"""

    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        self.model = MISTRAL_MODEL
        self.client = Mistral(api_key=MISTRAL_API_KEY)

    async def generate_completion_sync(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        """Implementation of non-streaming completion for Mistral"""
        try:
            # Set a default max_tokens if not provided to limit response size
            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = 2048
                
            response = await self.client.chat.complete_async(
                model=self.model,
                messages=messages,
                **kwargs
            )

            # Return response content and (placeholder) token usage
            # Note: Mistral client might not provide token usage like Anthropic does
            return {
                "content": response.choices[0].message.content,
                "input_tokens": 0,  # Placeholder - Mistral might not provide this
                "output_tokens": 0,  # Placeholder - Mistral might not provide this
            }
        except Exception as e:
            # Handle API errors gracefully
            error_message = f"Error generating completion: {str(e)}"
            return {
                "content": error_message,
                "input_tokens": 0,
                "output_tokens": 0,
            }


class TaskDecomposer:
    """Handles decomposition of queries into parallel tasks"""

    def __init__(self, llm_provider: MistralProvider):
        self.llm_provider = llm_provider
        self.prompt_template = DECOMPOSITION_PROMPT

    async def decompose_query(self, query: str, max_tasks: int = 4) -> Dict[str, Any]:
        """Decompose a query into multiple parallel tasks"""
        # Format the prompt with the user's query
        decomposition_prompt = self.prompt_template.format(user_query=query)

        # Call LLM for decomposition with a system message to ensure proper formatting
        response = await self.llm_provider.generate_completion_sync(
            [
                {"role": "system", "content": "You are a helpful task decomposition assistant. Follow the format exactly as requested."},
                {"role": "user", "content": decomposition_prompt}
            ]
        )

        decomposition_result = response["content"]

        # Extract token usage
        input_tokens = response.get("input_tokens", 0)
        output_tokens = response.get("output_tokens", 0)

        # Parse the decomposition result using regex
        decomposition_summary = re.search(
            r"DECOMPOSITION_SUMMARY:(.*?)(?:PARALLEL_TASKS_COUNT:|$)",
            decomposition_result,
            re.DOTALL,
        )
        tasks_count = re.search(r"PARALLEL_TASKS_COUNT:\s*(\d+)", decomposition_result)

        if not (decomposition_summary and tasks_count):
            return {
                "tasks": [{"subject": "Default", "prompt": query}],
                "summary": "Unable to decompose query",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        count = min(int(tasks_count.group(1)), max_tasks)  # Ensure we don't exceed max

        # Get each task subject and prompt
        tasks = []

        for i in range(1, count + 1):
            subject_pattern = f"TASK_{i}_SUBJECT:(.*?)(?:TASK_{i}_PROMPT:|$)"
            prompt_pattern = f"TASK_{i}_PROMPT:(.*?)(?:TASK_{i+1}_SUBJECT:|SYNTHESIS_RECOMMENDATION:|$)"

            subject_match = re.search(subject_pattern, decomposition_result, re.DOTALL)
            prompt_match = re.search(prompt_pattern, decomposition_result, re.DOTALL)

            if subject_match and prompt_match:
                subject = subject_match.group(1).strip()
                prompt = prompt_match.group(1).strip()

                tasks.append({"subject": subject, "prompt": prompt})

        # If we failed to get the right number of tasks, fall back to simpler approach
        if len(tasks) != count:
            return {
                "tasks": [{"subject": "Default", "prompt": query}],
                "summary": "Unable to properly decompose query",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        return {
            "tasks": tasks,
            "summary": decomposition_summary.group(1).strip()
            if decomposition_summary
            else "",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


class SynthesisGenerator:
    """Handles synthesis of parallel task results into a final response"""

    def __init__(self, llm_provider: MistralProvider):
        self.llm_provider = llm_provider
        self.prompt_template = SYNTHESIS_PROMPT

    async def generate_synthesis(
        self, user_query: str, task_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a synthesized response from multiple task results"""
        # Format the synthesis prompt
        task_results_text = ""
        
        # Ensure we don't create a prompt that's too long
        # Summarize task results if necessary to fit within token limits
        MAX_CONTENT_LENGTH = 2000  # Conservative character limit per task
        
        for i, result in enumerate(task_results):
            subject = result["subject"]
            content = result["content"]
            
            # Truncate long content if needed
            if len(content) > MAX_CONTENT_LENGTH:
                content = content[:MAX_CONTENT_LENGTH] + "... [Content truncated due to length]"
                
            task_results_text += f"RESULT {i+1} - {subject}:\n{content}\n\n"

        synthesis_prompt = self.prompt_template.format(
            user_query=user_query, task_results=task_results_text
        )

        # Call LLM for synthesis with strict token limits
        response = await self.llm_provider.generate_completion_sync(
            [
                {"role": "system", "content": "You are a synthesis assistant. Your response MUST be under 2000 characters to fit in a Discord message. Be extremely concise."},
                {"role": "user", "content": synthesis_prompt}
            ],
            max_tokens=1024  # Strict limit to ensure we don't exceed Discord's message limit
        )

        return response


class MistralAgent:
    def __init__(self):
        self.llm_provider = MistralProvider()
        self.task_decomposer = TaskDecomposer(self.llm_provider)
        self.synthesis_generator = SynthesisGenerator(self.llm_provider)

    async def process_task(self, task: Dict[str, str]) -> Dict[str, Any]:
        """Process a single task"""
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task["prompt"]},
            ]

            response = await self.llm_provider.generate_completion_sync(
                messages,
                max_tokens=1500  # Limit each task response as well
            )
            
            return {
                "subject": task["subject"],
                "content": response["content"],
                "input_tokens": response.get("input_tokens", 0),
                "output_tokens": response.get("output_tokens", 0),
            }
        except Exception as e:
            print(f"Error processing task '{task['subject']}': {str(e)}")
            # Return a placeholder response rather than failing the entire pipeline
            return {
                "subject": task["subject"],
                "content": f"Error processing this task: {str(e)}",
                "input_tokens": 0,
                "output_tokens": 0,
            }

    async def run(self, message: discord.Message):
        try:
            query = message.content
            print(f"Processing query: {query}")
            
            # For simple queries, skip decomposition
            if len(query.split()) < 15:
                print("Query is simple, skipping decomposition")
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT + " Keep your response under 2000 characters to fit in a Discord message."},
                    {"role": "user", "content": query},
                ]
                response = await self.llm_provider.generate_completion_sync(messages, max_tokens=1024)
                
                content = response["content"]
                # Check length
                if len(content) > 2000:
                    print(f"Warning: Direct response too long ({len(content)} chars). Truncating.")
                    content = content[:1997] + "..."
                    
                return content
            
            # 1. Decompose the query into tasks
            print("Decomposing query into tasks...")
            decomposition = await self.task_decomposer.decompose_query(query)
            
            # If decomposition failed, process directly
            if len(decomposition["tasks"]) == 1 and decomposition["tasks"][0]["subject"] == "Default":
                print("Decomposition failed or unnecessary, processing directly")
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT + " Keep your response under 2000 characters to fit in a Discord message."},
                    {"role": "user", "content": query},
                ]
                response = await self.llm_provider.generate_completion_sync(messages, max_tokens=1024)
                
                content = response["content"]
                # Check length
                if len(content) > 2000:
                    print(f"Warning: Fallback response too long ({len(content)} chars). Truncating.")
                    content = content[:1997] + "..."
                    
                return content
            
            # 2. Process tasks in parallel
            print(f"Processing {len(decomposition['tasks'])} tasks in parallel...")
            tasks = decomposition["tasks"]
            for i, task in enumerate(tasks):
                print(f"Task {i+1}: {task['subject']}")
            
            task_results = await asyncio.gather(
                *[self.process_task(task) for task in tasks]
            )
            
            # 3. Synthesize results
            print("Synthesizing results...")
            synthesis = await self.synthesis_generator.generate_synthesis(query, task_results)
            
            content = synthesis["content"]
            
            # Final check to ensure response is within Discord's limit
            if len(content) > 2000:
                print(f"Warning: Response too long ({len(content)} chars). Truncating to 2000 chars.")
                content = content[:1997] + "..."
                
            return content
        except Exception as e:
            error_message = f"An error occurred while processing your request: {str(e)}"
            print(f"Error in run method: {str(e)}")
            import traceback
            traceback.print_exc()
            return error_message
