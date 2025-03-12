import os
import re
import asyncio
from typing import List, Dict, Any, AsyncIterator, Optional
from mistralai import Mistral
import discord
import requests
import json
import aiohttp

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


class PerplexitySearchProvider:
    """Handles search queries to Perplexity Sonar Pro API"""
    
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self.base_url = "https://api.perplexity.ai/chat/completions"
        
    async def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search for literature related to the query using Perplexity Sonar Pro
        
        Args:
            query: The research direction or topic to search for
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing search results and metadata
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Create a search-optimized prompt
        search_prompt = f"Conduct a literature review on: {query}. Focus on peer-reviewed academic papers, research methodologies, key findings, and recent developments. Include relevant citations."
        
        data = {
            "model": "sonar-pro",  # Using Sonar model for literature search
            "messages": [
                {"role": "system", "content": "You are a research assistant conducting literature reviews. Provide comprehensive academic information with proper citations."},
                {"role": "user", "content": search_prompt}
            ],
            "temperature": 0.2,  # Lower temperature for more factual responses
            "max_tokens": 1500,
            "return_citations": True
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "results": result["choices"][0]["message"]["content"],
                            "citations": result.get("citations", []),
                            "query": query
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"Search failed with status {response.status}: {error_text}",
                            "query": query
                        }
        except Exception as e:
            return {
                "success": False,
                "error": f"Search request failed: {str(e)}",
                "query": query
            }


class ResearchFramer:
    """Frames user queries as research directions"""
    
    def __init__(self, llm_provider: MistralProvider):
        self.llm_provider = llm_provider
        
    async def frame_as_research_direction(self, user_query: str) -> Dict[str, Any]:
        """
        Transform a user query into a well-formed research direction
        
        Args:
            user_query: The original user message
            
        Returns:
            Dictionary with the framed research direction and metadata
        """
        framing_prompt = f"""
        Transform the following user query into a well-formed research direction:
        
        USER QUERY: {user_query}
        
        Your task:
        1. Identify the core research topic or question
        2. Reformulate it as an academic research direction
        3. Expand slightly to capture related important aspects
        4. Ensure it's specific enough for meaningful literature search
        5. Format it as a clear research direction statement
        
        RESEARCH DIRECTION:
        """
        
        response = await self.llm_provider.generate_completion_sync(
            [
                {"role": "system", "content": "You are a research assistant that helps frame informal queries as formal research directions."},
                {"role": "user", "content": framing_prompt}
            ],
            max_tokens=300
        )
        
        return {
            "original_query": user_query,
            "research_direction": response["content"].strip(),
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0)
        }


class MistralAgent:
    def __init__(self):
        self.llm_provider = MistralProvider()
        self.task_decomposer = TaskDecomposer(self.llm_provider)
        self.synthesis_generator = SynthesisGenerator(self.llm_provider)
        self.research_framer = ResearchFramer(self.llm_provider)
        self.search_provider = PerplexitySearchProvider()

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

    async def process_research_query(self, query: str) -> Dict[str, Any]:
        """
        Process a research query through the literature review pipeline
        
        Args:
            query: The user's research query
            
        Returns:
            Dictionary with search results and metadata
        """
        # Step 1: Frame the query as a research direction
        framed = await self.research_framer.frame_as_research_direction(query)
        
        # Step 2: Perform literature search
        search_results = await self.search_provider.search(framed["research_direction"])
        
        return {
            "original_query": query,
            "framed_direction": framed["research_direction"],
            "search_results": search_results,
            "success": search_results.get("success", False)
        }

    async def run(self, message: discord.Message):
        try:
            query = message.content
            print(f"Processing query: {query}")
            
            print("Using online literature review pipeline")
            
            # Process as research query
            results = await self.process_research_query(query)
            
            if results["success"]:
                response = f"**Research Direction:**\n{results['framed_direction']}\n\n"
                response += f"**Literature Review:**\n{results['search_results']['results']}"
                
                # Add citations if available
                if results['search_results'].get('citations'):
                    response += "\n\n**Sources:**\n"
                    for i, citation in enumerate(results['search_results']['citations'][:5]):
                        response += f"{i+1}. {citation}\n"
                        
                # Ensure response fits in Discord's message limit
                if len(response) > 2000:
                    response = response[:1997] + "..."
                    
                return response
            else:
                # Fall back to normal processing if literature search fails
                print(f"Literature search failed: {results['search_results'].get('error')}")
                pass  # Continue to standard processing below
            
            # The rest of the existing run method for non-research queries or fallback
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
