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
            # Remove default max_tokens setting
            response = await self.client.chat.complete_async(
                model=self.model, messages=messages, **kwargs
            )

            # Return response content and (placeholder) token usage
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
                {
                    "role": "system",
                    "content": "You are a helpful task decomposition assistant. Follow the format exactly as requested.",
                },
                {"role": "user", "content": decomposition_prompt},
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
                content = (
                    content[:MAX_CONTENT_LENGTH]
                    + "... [Content truncated due to length]"
                )

            task_results_text += f"RESULT {i+1} - {subject}:\n{content}\n\n"

        synthesis_prompt = self.prompt_template.format(
            user_query=user_query, task_results=task_results_text
        )

        # Call LLM for synthesis with strict token limits
        response = await self.llm_provider.generate_completion_sync(
            [
                {
                    "role": "system",
                    "content": "You are a synthesis assistant. Your response MUST be under 2000 characters to fit in a Discord message. Be extremely concise.",
                },
                {"role": "user", "content": synthesis_prompt},
            ],
            max_tokens=1024,  # Strict limit to ensure we don't exceed Discord's message limit
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
        if not self.api_key:
            print("❌ Perplexity API key is not set")
            return {
                "success": False,
                "error": "Perplexity API key not configured",
                "results": "",
                "citations": [],
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Create a more concise search-optimized prompt
        search_prompt = f"""Conduct a focused, concise literature review on: {query}
        
        Include:
        1. Key papers (authors, years)
        2. Major methodologies
        3. Significant findings
        4. Research gaps
        5. Future directions
        
        Be thorough but concise. Provide specific academic citations using [n] format."""

        data = {
            "model": "sonar-pro",  # Updated to a valid Perplexity model
            "messages": [
                {
                    "role": "system",
                    "content": "You are a research assistant conducting literature reviews. Be thorough but concise. Prioritize academic sources and provide proper citations.",
                },
                {"role": "user", "content": search_prompt},
            ],
            "temperature": 0.2,  # Lower temperature for more factual responses
            # No max_tokens - let it generate a complete response
        }

        try:
            print(f"🔍 Sending search request to Perplexity API for: {query[:50]}...")
            print(f"📝 Using model: {data['model']}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url, headers=headers, json=data
                ) as response:
                    response_text = await response.text()
                    print(f"📡 Perplexity API response status: {response.status}")

                    if response.status != 200:
                        print(f"❌ Error response: {response_text[:200]}...")
                        return {
                            "success": False,
                            "error": f"Search failed with status {response.status}: {response_text[:500]}",
                            "results": "",
                            "citations": [],
                        }

                    try:
                        response_json = json.loads(response_text)
                        print(f"📊 Response JSON keys: {list(response_json.keys())}")

                        # Extract content from the correct location in the response
                        # The Perplexity API returns the content in choices[0].message.content
                        if (
                            "choices" in response_json
                            and len(response_json["choices"]) > 0
                        ):
                            content = response_json["choices"][0]["message"]["content"]
                            print(
                                f"✅ Successfully extracted content: {len(content)} chars"
                            )

                            # Extract citations from the message content using regex
                            citations = []

                            # First check if API directly provides citations
                            if "citations" in response_json and isinstance(
                                response_json["citations"], list
                            ):
                                api_citations = response_json["citations"]
                                print(
                                    f"📚 API directly provided {len(api_citations)} citations"
                                )
                                if api_citations:
                                    citations = api_citations

                            # If API didn't provide citations, extract from content
                            if not citations:
                                # Look for citation markers in the text
                                citation_matches = re.findall(
                                    r"\[(\d+)\](.*?)(?=\n\[\d+\]|\n\n|$)",
                                    content,
                                    re.DOTALL,
                                )
                                if citation_matches:
                                    for match in citation_matches:
                                        # Include the citation number in brackets for authenticity
                                        citation_num = match[0]
                                        citation_text = match[1].strip()
                                        if citation_text:
                                            # Preserve the original format with citation number
                                            citations.append(
                                                f"[{citation_num}] {citation_text}"
                                            )

                                    print(
                                        f"📚 Extracted {len(citations)} citations with reference numbers"
                                    )
                                else:
                                    # Try to find a References/Bibliography section
                                    references_match = re.search(
                                        r"(?:References|Bibliography|Citations):\s*(.*?)(?=\n\n\n|$)",
                                        content,
                                        re.DOTALL,
                                    )
                                    if references_match:
                                        # Split by newlines to get individual citations
                                        references_text = references_match.group(1)
                                        ref_lines = references_text.strip().split("\n")
                                        for line in ref_lines:
                                            if line.strip():
                                                citations.append(line.strip())

                                        print(
                                            f"📚 Extracted {len(citations)} citations from references section"
                                        )

                            # Ensure we have at least some citations
                            if not citations:
                                # Last resort fallback
                                content_paragraphs = content.split("\n\n")
                                for para in content_paragraphs:
                                    if any(
                                        marker in para.lower()
                                        for marker in [
                                            "author",
                                            "journal",
                                            "conference",
                                            "proceedings",
                                            "university",
                                        ]
                                    ):
                                        if (
                                            len(para) > 20 and len(para) < 500
                                        ):  # Reasonable citation length
                                            citations.append(para.strip())
                                            if (
                                                len(citations) >= 5
                                            ):  # Limit to 5 fallback citations
                                                break

                                if citations:
                                    print(
                                        f"📚 Extracted {len(citations)} citation-like paragraphs as fallback"
                                    )

                            # Debug log
                            if citations:
                                print(f"📚 Final citation count: {len(citations)}")
                                print(f"Example citation: {citations[0][:100]}")

                            return {
                                "success": True,
                                "results": content,
                                "citations": citations,
                                "query": query,
                            }
                        else:
                            print("❌ No 'choices' found in response")
                            print(f"📄 Response: {response_text[:200]}...")
                            return {
                                "success": False,
                                "error": "Invalid response format from Perplexity API",
                                "results": "",
                                "citations": [],
                            }
                    except json.JSONDecodeError:
                        print("❌ Failed to parse JSON response")
                        return {
                            "success": False,
                            "error": f"Failed to parse JSON response: {response_text[:200]}...",
                            "results": "",
                            "citations": [],
                        }
        except Exception as e:
            error_message = f"Search request failed: {str(e)}"
            print(f"❌ Exception during search: {error_message}")
            import traceback

            traceback.print_exc()
            return {
                "success": False,
                "error": error_message,
                "results": "",
                "citations": [],
            }


class ResearchFramer:
    """Frames user queries as research directions"""

    def __init__(self, llm_provider: MistralProvider):
        self.llm_provider = llm_provider

    async def frame_as_research_direction(self, query: str) -> Dict[str, Any]:
        """
        Frame a user query as a formal research direction

        Args:
            query: The user's original query

        Returns:
            Dictionary with the framed research direction and metadata
        """
        try:
            # Create a more concise prompt
            framing_prompt = f"""Reframe this query as a clear, concise academic research direction:

Query: {query}

Return only the research direction without explanation or commentary."""

            # Get framed direction from LLM
            response = await self.llm_provider.generate_completion_sync(
                [
                    {
                        "role": "system",
                        "content": "You are a research question formulation assistant. Your job is to convert informal queries into precise, searchable research directions.",
                    },
                    {"role": "user", "content": framing_prompt},
                ]
            )

            # Extract framed direction
            research_direction = response["content"].strip()

            # If response is too long, truncate it
            if len(research_direction) > 150:
                research_direction = research_direction[:147] + "..."

            print(f"✅ Framed query as research direction: {research_direction}")

            return {
                "success": True,
                "research_direction": research_direction,
                "original_query": query,
                "input_tokens": response.get("input_tokens", 0),
                "output_tokens": response.get("output_tokens", 0),
            }
        except Exception as e:
            error_message = f"Error framing research direction: {str(e)}"
            print(f"❌ {error_message}")
            return {
                "success": False,
                "error": error_message,
                "research_direction": query,  # Fall back to original query
                "original_query": query,
            }


class DirectionReviewer:
    """Reviews and evaluates potential research directions"""

    def __init__(self, llm_provider: MistralProvider):
        self.llm_provider = llm_provider

    async def review_direction(
        self, direction: Dict[str, str], literature_context: str
    ) -> Dict[str, Any]:
        """
        Enhanced review of research direction with gap analysis and implementation pathway

        Args:
            direction: Dictionary containing the research direction details
            literature_context: Context from the literature review

        Returns:
            Dictionary with the review results
        """
        # Create a comprehensive review prompt
        review_prompt = f"""Evaluate this research direction against current literature:

DIRECTION: {direction.get('title', 'Untitled')}
QUESTION: {direction.get('question', 'No question provided')}
RATIONALE: {direction.get('rationale', 'No rationale provided')}
ADDRESSING GAP: {direction.get('gap_addressed', 'Not specified')}

LITERATURE CONTEXT:
{literature_context[:2000]}

REQUIRED ANALYSIS SECTIONS:

1. GAP VALIDATION:
- Is this truly a gap in current research?
- What evidence supports this?
- Have others attempted to address it?

2. TECHNICAL FEASIBILITY:
- Required capabilities
- Technical prerequisites
- Potential roadblocks
- Timeline estimate

3. RESEARCH IMPACT:
- Scientific significance
- Practical applications
- Broader implications
- Risk/reward assessment

4. RESOURCE REQUIREMENTS:
- Technical expertise needed
- Computing resources
- Data requirements
- Collaboration needs

5. COMPETITIVE ANALYSIS:
- Similar research efforts
- Unique advantages
- Potential competitors
- First-mover benefits

6. IMPLEMENTATION PATHWAY:
- Critical milestones
- Key experiments
- Validation methods
- Success metrics

PRIORITY RATING:
Score each category (1-5):
- Gap Significance: [score]
- Technical Feasibility: [score]
- Potential Impact: [score]
- Resource Availability: [score]
- Competitive Advantage: [score]

FINAL VERDICT:
[HIGH/MEDIUM/LOW] priority because [concise explanation]
"""

        # Get review from LLM with stronger formatting instructions
        response = await self.llm_provider.generate_completion_sync(
            [
                {
                    "role": "system",
                    "content": """You are a research direction evaluator specializing in:
1. Validating research gaps
2. Assessing technical feasibility
3. Evaluating potential impact
4. Resource planning
5. Competitive analysis

Provide concrete, specific assessments backed by the literature context.""",
                },
                {"role": "user", "content": review_prompt},
            ]
        )

        # Extract sections using regex
        review_content = response["content"]
        review = {
            "title": direction.get("title", "Untitled"),
            "question": direction.get("question", "No question provided"),
            "feasibility": self._extract_section(
                review_content, "TECHNICAL FEASIBILITY"
            ),
            "novelty": self._extract_section(review_content, "GAP VALIDATION"),
            "impact": self._extract_section(review_content, "RESEARCH IMPACT"),
            "pros": self._extract_section(review_content, "UNIQUE ADVANTAGES"),
            "cons": self._extract_section(review_content, "POTENTIAL ROADBLOCKS"),
            "resources": self._extract_section(review_content, "RESOURCE REQUIREMENTS"),
            "priority": self._extract_section(review_content, "FINAL VERDICT"),
        }

        return review

    async def compare_directions(
        self, directions: List[Dict[str, str]], literature_context: str
    ) -> Dict[str, Any]:
        """
        Compare multiple research directions in parallel

        Args:
            directions: List of research directions to compare
            literature_context: Context from the literature review

        Returns:
            Dictionary with detailed reviews and comparison
        """
        if not directions:
            return {
                "success": False,
                "error": "No research directions provided to review",
                "reviews": [],
            }

        print(f"🔍 Reviewing {len(directions)} research directions in parallel...")

        # Process each direction review in parallel
        review_tasks = [
            self.review_direction(direction, literature_context)
            for direction in directions
        ]
        reviews = await asyncio.gather(*review_tasks)

        print("✅ Direction reviews completed")

        # Generate a comparison of the directions
        comparison = await self._generate_comparison(directions, reviews)

        return {
            "success": True,
            "reviews": reviews,
            "comparison": comparison,
            "count": len(reviews),
        }

    def _format_reviews_summary(self, reviews: List[Dict[str, str]]) -> str:
        """Format a summary of reviews for comparison"""
        summary = ""

        for i, review in enumerate(reviews):
            title = review.get("title", f"Direction {i+1}")
            summary += f"DIRECTION {i+1}: {title}\n"

            # Add important sections
            if "novelty" in review:
                summary += f"Gap Analysis: {review['novelty'][:150]}...\n"

            if "feasibility" in review:
                summary += f"Technical Feasibility: {review['feasibility'][:150]}...\n"

            if "impact" in review:
                summary += f"Impact: {review['impact'][:150]}...\n"

            if "priority" in review:
                summary += f"Priority: {review['priority']}\n"

            summary += "\n"

        return summary

    async def _generate_comparison(
        self, directions: List[Dict[str, str]], reviews: List[Dict[str, str]]
    ) -> str:
        """Generate a strategic comparison of research directions"""

        comparison_prompt = f"""Compare these evaluated research directions:

{self._format_reviews_summary(reviews)}

Provide analysis in these sections:

1. STRATEGIC RANKING:
- Rank directions by overall promise
- Explain ranking rationale
- Note key differentiators

2. RESOURCE OPTIMIZATION:
- Identify shared resources
- Note potential synergies
- Suggest parallel efforts

3. RISK PORTFOLIO:
- Balance of risk/reward
- Diversification strategy
- Fallback options

4. EXECUTION STRATEGY:
- Recommended sequence
- Critical dependencies
- Early validation points

5. FINAL RECOMMENDATION:
- Primary direction
- Supporting directions
- Key success factors

Keep response under 2000 chars. Be specific and actionable."""

        response = await self.llm_provider.generate_completion_sync(
            [
                {
                    "role": "system",
                    "content": "You are a research advisor who compares different research directions and provides strategic recommendations.",
                },
                {"role": "user", "content": comparison_prompt},
            ],
            max_tokens=1000,
        )

        return response["content"]

    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a specific section from the structured content with better handling of variations"""
        # Try exact format first (SECTION_NAME:)
        pattern = f"{section_name}:(.*?)(?:(?:[A-Z_]+:)|$)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try variations with flexible capitalization and optional underscore/space
        # This will match "Next Steps:", "NEXT STEPS:", "next_steps:", etc.
        section_pattern = section_name.replace("_", "[_ ]?")
        pattern = f"(?i){section_pattern}:(.*?)(?:(?:[A-Za-z _]+:)|$)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try looking for section with ## or other markdown headers
        pattern = f"(?i)(?:##\\s*|\\*\\*\\s*){section_pattern.replace('_', '[_ ]?')}(?:\\s*:|\\s*\\*\\*)?\\s*(.*?)(?:(?:##|\\*\\*)[A-Za-z _]+|$)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # If all else fails, return empty string
        return ""


class ResearchDirectionsGenerator:
    """Generates potential research directions based on literature review results"""

    def __init__(self, llm_provider: MistralProvider):
        self.llm_provider = llm_provider

    async def generate_directions(
        self, search_results: Dict[str, Any], count: int = 3
    ) -> Dict[str, Any]:
        """
        Generate multiple potential research directions based on literature review

        Args:
            search_results: Results from the literature search
            count: Number of research directions to generate (default: 3)

        Returns:
            Dictionary with generated research directions and metadata
        """
        if not search_results.get("success", False):
            print("❌ Cannot generate directions: Search was not successful")
            return {
                "success": False,
                "error": "Cannot generate research directions without successful literature search",
                "directions": [],
            }

        # Extract the content from search results
        content = search_results.get("results", "")

        if not content or len(content) < 100:
            print(f"❌ Search results content is too short: {len(content)} chars")
            sample = content[:50] + "..." if content else "[empty]"
            print(f"📄 Content sample: {sample}")
            return {
                "success": False,
                "error": "Search results content is insufficient for generating directions",
                "directions": [],
            }

        print(f"📝 Generating directions from {len(content)} chars of content")

        # Create a prompt with clearer formatting instructions
        generation_prompt = f"""
        Based on the following literature review, generate {count} potential research directions or hypotheses:
        
        LITERATURE REVIEW:
        {content[:4000]}
        
        REQUIRED OUTPUT FORMAT:
        You must format your response exactly as shown below, using these exact headings:
        
        DIRECTION_1_TITLE: [Brief title for the research direction]
        DIRECTION_1_QUESTION: [The specific research question]
        DIRECTION_1_RATIONALE: [Why this direction is promising]
        DIRECTION_1_METHODOLOGY: [Potential research methodologies]
        DIRECTION_1_GAPS: [Gaps this addresses]
        
        DIRECTION_2_TITLE: [Brief title for the research direction]
        DIRECTION_2_QUESTION: [The specific research question]
        ...
        
        Continue this format for all {count} directions. Do not include any other sections or explanatory text.
        """

        try:
            # Get directions from LLM with stronger formatting instructions
            response = await self.llm_provider.generate_completion_sync(
                [
                    {
                        "role": "system",
                        "content": """You are a research direction generator who identifies promising research opportunities based on literature reviews.

You MUST follow the output format instructions precisely, using the exact section headers requested with the exact capitalization and underscores shown.""",
                    },
                    {"role": "user", "content": generation_prompt},
                ]
            )

            # Parse the response to extract research directions
            generated_content = response["content"]
            print(f"✅ Received {len(generated_content)} chars of generated content")

            # Debug: show a preview of the generated content
            preview = generated_content[:300].replace("\n", " ")
            print(f"📄 Content preview: {preview}...")

            directions = []

            # Extract each direction using regex
            for i in range(1, count + 1):
                direction = {}

                # Extract title
                title_pattern = (
                    f"DIRECTION_{i}_TITLE:(.*?)(?:DIRECTION_{i}_QUESTION:|$)"
                )
                title_match = re.search(title_pattern, generated_content, re.DOTALL)
                if title_match:
                    direction["title"] = title_match.group(1).strip()
                    print(f"✅ Found direction {i} title: {direction['title']}")
                else:
                    print(f"❓ Missing title for direction {i}")

                # Extract question (only if title was found)
                if "title" in direction:
                    question_pattern = (
                        f"DIRECTION_{i}_QUESTION:(.*?)(?:DIRECTION_{i}_RATIONALE:|$)"
                    )
                    question_match = re.search(
                        question_pattern, generated_content, re.DOTALL
                    )
                    if question_match:
                        direction["question"] = question_match.group(1).strip()

                    # Extract rationale
                    rationale_pattern = (
                        f"DIRECTION_{i}_RATIONALE:(.*?)(?:DIRECTION_{i}_METHODOLOGY:|$)"
                    )
                    rationale_match = re.search(
                        rationale_pattern, generated_content, re.DOTALL
                    )
                    if rationale_match:
                        direction["rationale"] = rationale_match.group(1).strip()

                    # Extract methodology
                    methodology_pattern = f"DIRECTION_{i}_METHODOLOGY:(.*?)(?:DIRECTION_{i}_GAPS:|DIRECTION_{i+1}_TITLE:|$)"
                    methodology_match = re.search(
                        methodology_pattern, generated_content, re.DOTALL
                    )
                    if methodology_match:
                        direction["methodology"] = methodology_match.group(1).strip()

                    # Extract gaps
                    gaps_pattern = (
                        f"DIRECTION_{i}_GAPS:(.*?)(?:DIRECTION_{i+1}_TITLE:|$)"
                    )
                    gaps_match = re.search(gaps_pattern, generated_content, re.DOTALL)
                    if gaps_match:
                        direction["gaps"] = gaps_match.group(1).strip()

                # Add to directions if we have at least a title and question
                if direction.get("title") and direction.get("question"):
                    directions.append(direction)

            print(f"💡 Successfully extracted {len(directions)} research directions")

            # If no directions were found with regex, use a fallback approach
            if not directions:
                print("⚠️ No directions found with regex, trying fallback parsing...")
                lines = generated_content.split("\n")
                current_direction = {}
                current_field = None

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Check for a new direction title
                    if line.startswith("DIRECTION") and "TITLE:" in line:
                        # Save previous direction if it exists
                        if current_direction.get("title") and current_direction.get(
                            "question"
                        ):
                            directions.append(current_direction)
                            print(
                                f"✅ Added direction from fallback: {current_direction['title']}"
                            )

                        current_direction = {}
                        parts = line.split("TITLE:", 1)
                        if len(parts) > 1:
                            current_direction["title"] = parts[1].strip()
                            current_field = "title"

                    # Check for other fields
                    elif "QUESTION:" in line:
                        parts = line.split("QUESTION:", 1)
                        if len(parts) > 1:
                            current_direction["question"] = parts[1].strip()
                            current_field = "question"
                    elif "RATIONALE:" in line:
                        parts = line.split("RATIONALE:", 1)
                        if len(parts) > 1:
                            current_direction["rationale"] = parts[1].strip()
                            current_field = "rationale"
                    elif "METHODOLOGY:" in line:
                        parts = line.split("METHODOLOGY:", 1)
                        if len(parts) > 1:
                            current_direction["methodology"] = parts[1].strip()
                            current_field = "methodology"
                    elif "GAPS:" in line:
                        parts = line.split("GAPS:", 1)
                        if len(parts) > 1:
                            current_direction["gaps"] = parts[1].strip()
                            current_field = "gaps"
                    # Append to current field if continuing
                    elif current_field and current_field in current_direction:
                        current_direction[current_field] += " " + line

                # Add the last direction if it exists
                if current_direction.get("title") and current_direction.get("question"):
                    directions.append(current_direction)
                    print(
                        f"✅ Added final direction from fallback: {current_direction['title']}"
                    )

            print(f"💡 Final count: {len(directions)} research directions")

            return {
                "success": len(directions) > 0,
                "directions": directions,
                "count": len(directions),
                "input_tokens": response.get("input_tokens", 0),
                "output_tokens": response.get("output_tokens", 0),
            }
        except Exception as e:
            error_message = f"Error generating research directions: {str(e)}"
            print(f"❌ {error_message}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": error_message, "directions": []}


class RecommendationSynthesizer:
    """Synthesizes final research recommendations and next steps"""

    def __init__(self, llm_provider: MistralProvider):
        self.llm_provider = llm_provider

    def _build_full_synthesis_prompt(
        self, query: str, lit_review: str, comparison: str
    ) -> str:
        """Build synthesis prompt when we have both literature review and direction reviews"""
        return f"""
        Based on the following research information, provide clear research recommendations and next steps:
        
        ORIGINAL QUERY: {query}
        
        LITERATURE REVIEW SUMMARY:
        {lit_review[:1500]}
        
        DIRECTION COMPARISON:
        {comparison[:1000]}
        
        REQUIRED OUTPUT FORMAT - YOU MUST USE THESE EXACT SECTION HEADERS WITH COLONS:

        RECOMMENDATIONS: [Provide 2-3 synthesized research recommendations based on the literature and comparisons. Be specific and actionable.]
        
        NEXT_STEPS: [List 3-5 clear, numbered steps the researcher should take, ordered by priority. Format as "1. First step", "2. Second step", etc.]
        
        TIMELINE: [Suggest a realistic timeline for implementing these steps.]
        
        POTENTIAL_CHALLENGES: [Note 2-3 challenges the researcher might face and brief suggestions to overcome them.]
        
        Do not include any other sections, and make sure to use the exact section headers as shown above (all caps followed by colon).
        """

    def _build_directions_only_prompt(
        self, query: str, lit_review: str, directions: List[Dict[str, str]]
    ) -> str:
        """Build synthesis prompt when we have literature review and directions but no reviews"""
        directions_text = "\n\n".join(
            [
                f"DIRECTION {i+1}: {direction.get('title', 'Untitled')}\n{direction.get('question', 'No question provided')}"
                for i, direction in enumerate(directions[:3])
            ]
        )

        return f"""
        Based on the following research information, provide clear research recommendations and next steps:
        
        ORIGINAL QUERY: {query}
        
        LITERATURE REVIEW SUMMARY:
        {lit_review[:1500]}
        
        POTENTIAL RESEARCH DIRECTIONS:
        {directions_text}
        
        REQUIRED OUTPUT FORMAT - YOU MUST USE THESE EXACT SECTION HEADERS WITH COLONS:

        RECOMMENDATIONS: [Provide 2-3 synthesized research recommendations based on the literature and potential directions. Be specific and actionable.]
        
        NEXT_STEPS: [List 3-5 clear, numbered steps the researcher should take, ordered by priority. Format as "1. First step", "2. Second step", etc.]
        
        TIMELINE: [Suggest a realistic timeline for implementing these steps.]
        
        POTENTIAL_CHALLENGES: [Note 2-3 challenges the researcher might face and brief suggestions to overcome them.]
        
        Do not include any other sections, and make sure to use the exact section headers as shown above (all caps followed by colon).
        """

    def _build_literature_only_prompt(self, query: str, lit_review: str) -> str:
        """Build synthesis prompt when we only have literature review"""
        return f"""
        Based on the following literature review, provide research recommendations and next steps:
        
        ORIGINAL QUERY: {query}
        
        LITERATURE REVIEW:
        {lit_review}
        
        REQUIRED OUTPUT FORMAT - YOU MUST USE THESE EXACT SECTION HEADERS WITH COLONS:

        RECOMMENDATIONS: [Provide 2-3 promising research directions based solely on the literature review. For each, explain the rationale and how it addresses gaps in existing research.]
        
        NEXT_STEPS: [List 3-5 clear, numbered steps the researcher should take, ordered by priority. Format as "1. First step", "2. Second step", etc.]
        
        TIMELINE: [Suggest a realistic timeline for implementing these steps.]
        
        POTENTIAL_CHALLENGES: [Note 2-3 challenges the researcher might face and brief suggestions to overcome them.]
        
        Do not include any other sections, and make sure to use the exact section headers as shown above (all caps followed by colon).
        """

    async def synthesize_recommendations(
        self,
        query: str,
        search_results: Dict[str, Any],
        directions_result: Dict[str, Any],
        reviews_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate synthesized recommendations and next steps"""
        print("🔄 Synthesizing final recommendations...")

        try:
            # Check if we have valid inputs
            if not search_results.get("success", False):
                print("❌ Cannot synthesize recommendations: Search was not successful")
                return {
                    "success": False,
                    "error": "Cannot synthesize recommendations without successful literature search",
                    "recommendations": "No recommendations available due to search failure.",
                    "next_steps": [],
                }

            # Extract the necessary information from inputs
            lit_review = search_results.get("results", "")[
                :2000
            ]  # Limit to avoid token overflow
            citations = search_results.get("citations", [])

            # Build different prompts based on available data
            if reviews_result.get("success", False) and reviews_result.get(
                "comparison", ""
            ):
                # We have both directions and reviews
                comparison = reviews_result.get("comparison", "")
                prompt = self._build_full_synthesis_prompt(
                    query, lit_review, comparison
                )
            elif directions_result.get("success", False) and directions_result.get(
                "directions", []
            ):
                # We have directions but no reviews
                directions = directions_result.get("directions", [])
                prompt = self._build_directions_only_prompt(
                    query, lit_review, directions
                )
            else:
                # We only have literature review
                prompt = self._build_literature_only_prompt(query, lit_review)

            # Get synthesis from LLM with stronger formatting instructions
            response = await self.llm_provider.generate_completion_sync(
                [
                    {
                        "role": "system",
                        "content": """You are a research advisor who synthesizes research findings and provides actionable recommendations.

You MUST follow the output format instructions precisely. Your response should ALWAYS include the exact section headers requested (RECOMMENDATIONS:, NEXT_STEPS:, etc.) with no variations. 

Format next steps as numbered lists (1., 2., etc.) and ensure each section has substantive content.""",
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            # Process the response
            content = response["content"]

            # Add logging to debug empty response
            print(f"📄 Synthesis response length: {len(content)} chars")
            if len(content) < 50:
                print(f"⚠️ Very short synthesis response: '{content}'")
                # Return error status for empty response
                return {
                    "success": False,
                    "error": "Synthesis generated empty or too short response",
                    "recommendations": "Could not generate detailed recommendations. Please try refining your query.",
                    "next_steps": [
                        {
                            "number": "1",
                            "description": "Try a more specific research query",
                        }
                    ],
                    "citations": [],
                }

            # Extract next steps section
            next_steps_section = self._extract_section(content, "NEXT_STEPS")
            next_steps = self._parse_steps(next_steps_section)

            # Extract recommendations section
            recommendations = self._extract_section(content, "RECOMMENDATIONS")

            # Ensure recommendations are not empty
            if not recommendations or len(recommendations.strip()) < 20:
                print("⚠️ Empty recommendations extracted from non-empty synthesis")
                # Extract any section that might have content
                for section_name in [
                    "SYNTHESIS",
                    "SUMMARY",
                    "OVERVIEW",
                    "ANALYSIS",
                    "RESULTS",
                ]:
                    alt_section = self._extract_section(content, section_name)
                    if alt_section and len(alt_section.strip()) >= 20:
                        recommendations = alt_section
                        break

                # If still empty, use the first 500 chars of the response
                if not recommendations or len(recommendations.strip()) < 20:
                    recommendations = content[:500] + "..."

            # Ensure next_steps are not empty
            if not next_steps:
                print("⚠️ Empty next steps extracted from non-empty synthesis")
                next_steps = [
                    {
                        "number": "1",
                        "description": "Conduct a more focused literature review on the specific aspects identified above",
                    },
                    {
                        "number": "2",
                        "description": "Consider reaching out to experts in the field for additional insights",
                    },
                ]

            print(
                f"✅ Successfully synthesized recommendations ({len(recommendations)} chars) and {len(next_steps)} next steps"
            )

            return {
                "success": True,
                "recommendations": recommendations,
                "next_steps": next_steps,
                "full_synthesis": content,
                "citations": citations,
            }

        except Exception as e:
            error_message = f"Error synthesizing recommendations: {str(e)}"
            print(f"❌ {error_message}")
            import traceback

            traceback.print_exc()

            # Return a fallback response instead of empty content
            return {
                "success": False,
                "error": error_message,
                "recommendations": "Could not generate recommendations due to an error. Please try again with a different query.",
                "next_steps": [
                    {"number": "1", "description": "Try a more specific research query"}
                ],
                "citations": [],
            }

    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a specific section from the structured content with better handling of variations"""
        # Try exact format first (SECTION_NAME:)
        pattern = f"{section_name}:(.*?)(?:(?:[A-Z_]+:)|$)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try variations with flexible capitalization and optional underscore/space
        # This will match "Next Steps:", "NEXT STEPS:", "next_steps:", etc.
        section_pattern = section_name.replace("_", "[_ ]?")
        pattern = f"(?i){section_pattern}:(.*?)(?:(?:[A-Za-z _]+:)|$)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try looking for section with ## or other markdown headers
        pattern = f"(?i)(?:##\\s*|\\*\\*\\s*){section_pattern.replace('_', '[_ ]?')}(?:\\s*:|\\s*\\*\\*)?\\s*(.*?)(?:(?:##|\\*\\*)[A-Za-z _]+|$)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # If all else fails, return empty string
        return ""

    def _parse_steps(self, steps_text: str) -> List[Dict[str, str]]:
        """Parse the next steps text into a structured list with better resilience"""
        steps = []

        if not steps_text or len(steps_text.strip()) < 10:
            # Return default steps if text is too short
            return [
                {
                    "number": "1",
                    "description": "Review the literature findings and select a specific focus area",
                },
                {
                    "number": "2",
                    "description": "Develop a detailed research plan based on the recommendations",
                },
            ]

        # Look for numbered steps (1., 2., etc.)
        step_matches = re.findall(
            r"(\d+[\.\)\-])\s+(.*?)(?=\n\s*\d+[\.\)\-]|\n\n|$)", steps_text, re.DOTALL
        )

        if step_matches:
            for match in step_matches:
                number, text = match
                steps.append(
                    {"number": number.strip(".)-"), "description": text.strip()}
                )
        else:
            # Alternative parsing for bullet points or other formats
            lines = steps_text.split("\n")
            current_step = None
            step_number = 1

            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue

                # Check if this looks like a new step
                if (
                    stripped.startswith("•")
                    or stripped.startswith("-")
                    or stripped.startswith("*")
                    or (
                        len(stripped) > 2
                        and stripped[0].isdigit()
                        and stripped[1] in [".", ")", ":", "-"]
                    )
                ):
                    if current_step:
                        steps.append(current_step)
                    current_step = {
                        "number": str(step_number),
                        "description": stripped.lstrip("•-*123456789.) :"),
                    }
                    step_number += 1
                elif current_step:
                    # Continue previous step
                    current_step["description"] += " " + stripped

            # Add the last step if exists
            if current_step:
                steps.append(current_step)

        # If no steps were found but we have content, break it into artificial steps
        if not steps and steps_text:
            # Split by sentences or paragraphs
            parts = re.split(r"(?<=[.!?])\s+", steps_text)
            for i, part in enumerate(parts[:3]):  # Limit to 3 steps
                if len(part.strip()) > 10:  # Only add if meaningful content
                    steps.append({"number": str(i + 1), "description": part.strip()})

        # If still empty, add a default step
        if not steps:
            steps.append(
                {
                    "number": "1",
                    "description": "Review the literature and synthesize key findings into a research plan",
                }
            )

        return steps


class MistralAgent:
    def __init__(self):
        self.llm_provider = MistralProvider()
        self.task_decomposer = TaskDecomposer(self.llm_provider)
        self.synthesis_generator = SynthesisGenerator(self.llm_provider)
        self.research_framer = ResearchFramer(self.llm_provider)
        self.search_provider = PerplexitySearchProvider()
        self.directions_generator = ResearchDirectionsGenerator(self.llm_provider)
        self.direction_reviewer = DirectionReviewer(self.llm_provider)
        self.recommendation_synthesizer = RecommendationSynthesizer(self.llm_provider)

    def _format_citation_list(self, citations, max_display=5):
        """Helper function to format citations consistently across the application"""
        formatted_content = ""

        # Just number the citations - no additional formatting or cleaning
        for i, citation in enumerate(citations[:max_display]):
            citation_num = i + 1

            # Just trim whitespace but don't modify content
            citation_text = citation.strip()

            # Add a zero-width space after any http/https to prevent Discord from creating previews
            # This makes links still clickable but doesn't create embeds
            citation_text = citation_text.replace(
                "http://", "http://​"
            )  # Zero-width space after //
            citation_text = citation_text.replace(
                "https://", "https://​"
            )  # Zero-width space after //

            # Format as markdown numbered list item
            formatted_content += f"{citation_num}. {citation_text}\n"

        if len(citations) > max_display:
            formatted_content += (
                f"*(plus {len(citations) - max_display} more sources)*\n"
            )

        return formatted_content

    async def process_task(self, task: Dict[str, str]) -> Dict[str, Any]:
        """Process a single task"""
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task["prompt"]},
            ]

            response = await self.llm_provider.generate_completion_sync(
                messages,
                max_tokens=1500,  # Limit each task response as well
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
        try:
            # Step 1: Frame the query as a research direction
            print("📚 Step 1: Framing research direction...")
            framed = await self.research_framer.frame_as_research_direction(query)
            framed_direction = framed["research_direction"]
            print(f"🎯 Framed Direction: {framed_direction}")

            # Step 2: Perform literature search
            print("🔍 Step 2: Conducting literature search...")
            search_results = await self.search_provider.search(framed_direction)

            if search_results.get("success", False):
                print("📖 Literature search successful")
                # Log a bit of the search results for debugging
                print(
                    f"📄 Search results preview: {search_results.get('results', '')[:200]}..."
                )

                # Step 3: Generate potential research directions
                print("🧠 Step 3: Generating potential research directions...")
                directions_result = await self.directions_generator.generate_directions(
                    search_results
                )
                print(
                    f"💡 Generated {directions_result.get('count', 0)} research directions"
                )

                # Step 4: Review research directions
                reviews_result = {}
                if (
                    directions_result.get("success", False)
                    and directions_result.get("count", 0) > 0
                ):
                    print("⚖️ Step 4: Evaluating research directions...")
                    reviews_result = await self.direction_reviewer.compare_directions(
                        directions_result["directions"], search_results["results"]
                    )
                    print("✅ Direction evaluations completed")
                else:
                    print(
                        "⚠️ Skipping direction evaluation - no valid directions generated"
                    )

                # Step 5: Synthesize final recommendations
                print("🧠 Step 5: Synthesizing final recommendations...")
                synthesis_result = (
                    await self.recommendation_synthesizer.synthesize_recommendations(
                        query, search_results, directions_result, reviews_result
                    )
                )
                print("✅ Recommendation synthesis completed")

                # Create the final response
                response = f"## Research on: {framed_direction}\n\n"

                if synthesis_result.get("success", False):
                    response += f"### Recommendations\n{synthesis_result.get('recommendations', 'No specific recommendations available.')}\n\n"

                    # Add next steps
                    if synthesis_result.get("next_steps", []):
                        response += "### Next Steps\n"
                        for step in synthesis_result.get("next_steps", []):
                            response += f"{step.get('number', '•')}. {step.get('description', '')}\n"
                        response += "\n"

                    # Add citations if available
                    citations = synthesis_result.get("citations", [])
                    if citations:
                        response += "### Recommended Reading\nThese sources provide additional context for your research:\n\n"
                        for i, citation in enumerate(
                            citations[:5]
                        ):  # Limit to 5 citations to avoid too long responses
                            response += f"{i+1}. {citation}\n"
                        response += "\n"
                        if len(citations) > 5:
                            response += f"*Plus {len(citations) - 5} more sources not shown here.*\n\n"
                else:
                    # Fallback if synthesis failed
                    response += f"I was able to find information on your query, but had trouble synthesizing recommendations.\n\n"
                    response += f"Here's a summary of what I found:\n\n"

                    # Include a preview of search results
                    results_preview = (
                        search_results.get("results", "")[:500] + "..."
                        if len(search_results.get("results", "")) > 500
                        else search_results.get("results", "")
                    )
                    response += f"{results_preview}\n\n"

                    # Add basic next steps
                    response += "### Suggested Next Steps\n"
                    response += "1. Review the literature summary above\n"
                    response += "2. Consider narrowing your research focus\n"
                    response += "3. Try a follow-up query with 'refine:' prefix\n"

                return {
                    "original_query": query,
                    "framed_direction": framed_direction,
                    "search_results": search_results,
                    "directions_result": directions_result,
                    "reviews_result": reviews_result,
                    "synthesis_result": synthesis_result,
                    "success": True,
                }
            else:
                print(
                    f"❌ Literature search failed: {search_results.get('error', 'Unknown error')}"
                )
                return {
                    "original_query": query,
                    "framed_direction": framed_direction,
                    "search_results": search_results,
                    "success": False,
                }
        except Exception as e:
            error_message = f"Error in research query processing: {str(e)}"
            print(f"❌ {error_message}")
            import traceback

            traceback.print_exc()
            return {"original_query": query, "success": False, "error": error_message}

    async def run(self, message: discord.Message):
        try:
            query = message.content
            print(f"🤖 Starting to process query: {query}")

            # First, look for and delete any existing research status messages from the bot in this channel
            # to prevent duplicate status messages from previous interrupted runs
            try:
                async for old_message in message.channel.history(limit=10):
                    if old_message.author == message.guild.me and (
                        old_message.content.startswith("# Research Progress")
                        or "Step 1/5:" in old_message.content
                        or "The process can take a few minutes" in old_message.content
                        or "Starting research process" in old_message.content
                    ):
                        print(
                            f"🧹 Cleaning up previous status message: {old_message.id}"
                        )
                        await old_message.delete()
                        await asyncio.sleep(
                            0.5
                        )  # Small delay to ensure Discord API can process the deletion
            except Exception as e:
                print(f"Warning: Could not clean up old status messages: {e}")

            # Check if this is a refinement of a previous query
            is_refinement = False
            original_query = query

            # Check if message starts with "refine:" or similar refinement indicators
            refinement_prefixes = ["refine:", "iterate:", "narrow:", "focus:"]
            for prefix in refinement_prefixes:
                if query.lower().startswith(prefix):
                    is_refinement = True
                    query = query[len(prefix) :].strip()
                    print(f"📌 Detected refinement request: {query}")
                    break

            # Create a lock for message updates to avoid race conditions
            self.status_lock = getattr(self, "status_lock", None)
            if self.status_lock is None:
                self.status_lock = asyncio.Lock()

            # Initial status message content
            status_content = f'# Research Progress\n*Researching: "{query}"*\n\n'
            status_content += (
                "📚 **Step 1/5:** Framing your query as a research direction...\n"
            )

            # Send the initial status message
            status_message = await message.channel.send(status_content)

            try:
                # Function to safely update the status message
                async def update_status(new_content):
                    async with self.status_lock:
                        try:
                            # Check if message still exists and is accessible
                            try:
                                # Try to fetch the message to verify it exists
                                # This is a lightweight operation that won't hit rate limits
                                check_message = await message.channel.fetch_message(
                                    status_message.id
                                )
                                if check_message:
                                    await status_message.edit(content=new_content)
                            except discord.NotFound:
                                print(
                                    "Warning: Status message no longer exists, cannot update"
                                )
                                return
                            except discord.HTTPException as http_err:
                                if "Unknown Message" in str(http_err):
                                    print(
                                        "Warning: Status message no longer exists (404 Unknown Message)"
                                    )
                                    return
                                else:
                                    # Other HTTP errors - could be rate limits, etc.
                                    print(
                                        f"Warning: HTTP error when checking message: {http_err}"
                                    )

                            # Small delay to ensure Discord API rate limits aren't hit
                            await asyncio.sleep(0.5)
                        except Exception as e:
                            print(f"Warning: Failed to update status message: {e}")
                            if "Unknown Message" in str(e):
                                print(
                                    "Status message no longer exists - this is expected if the message was deleted"
                                )

                # Step 1: Frame the query as a research direction
                print("📚 Step 1: Framing research direction...")
                try:
                    framed = await self.research_framer.frame_as_research_direction(
                        query
                    )
                    if not framed.get("success", False):
                        error_msg = framed.get(
                            "error", "Unknown error in research framing"
                        )
                        print(f"❌ Research framing failed: {error_msg}")

                        status_content = (
                            f'# Research Progress\n*Researching: "{query}"*\n\n'
                        )
                        status_content += (
                            f"📚 **Step 1/5:** ❌ Failed to frame research direction\n"
                        )
                        status_content += f"*Error:* {error_msg[:200]}...\n\n"
                        status_content += "⚠️ Generating a general response instead..."
                        await update_status(status_content)

                        # Generate a fallback response
                        response = f"I wasn't able to properly frame your query as a research direction. The error was: {error_msg}\n\n"
                        response += "You might want to try:\n"
                        response += "1. Rephrasing your query to be more specific\n"
                        response += (
                            "2. Breaking your question into smaller, focused parts\n"
                        )
                        response += "3. Using more academic or scientific terminology\n"

                        # Send the response first
                        await message.channel.send(response)

                        # Safely delete the status message
                        try:
                            await status_message.delete()
                        except (discord.NotFound, discord.HTTPException) as e:
                            print(f"Status message deletion failed: {e}")

                        return None  # Return None to prevent duplicate response

                    framed_direction = framed["research_direction"]
                    print(f"🎯 Framed Direction: {framed_direction}")

                    # Update status message with framed direction
                    status_content = (
                        f'# Research Progress\n*Researching: "{query}"*\n\n'
                    )
                    status_content += (
                        f"📚 **Step 1/5:** ✅ Query framed as research direction\n"
                    )
                    status_content += f"*Research direction: **{framed_direction}**\n\n"
                    status_content += (
                        "🔍 **Step 2/5:** Searching scientific literature...\n"
                    )
                    await update_status(status_content)

                    # Step 2: Perform literature search
                    print("🔍 Step 2: Conducting literature search...")
                    search_results = await self.search_provider.search(framed_direction)
                except Exception as e:
                    error_msg = f"Error in research framing step: {str(e)}"
                    print(f"❌ {error_msg}")

                    status_content = (
                        f'# Research Progress\n*Researching: "{query}"*\n\n'
                    )
                    status_content += (
                        f"📚 **Step 1/5:** ❌ Failed to frame research direction\n"
                    )
                    status_content += f"*Error:* {str(e)[:200]}...\n\n"
                    status_content += "⚠️ Process terminated due to error."
                    await update_status(status_content)

                    # Generate error response
                    response = f"An error occurred while framing your query as a research direction: {str(e)}\n\nPlease try again with a different query."
                    await message.channel.send(response)

                    # Safely delete the status message
                    try:
                        await status_message.delete()
                    except (discord.NotFound, discord.HTTPException) as e:
                        print(f"Status message deletion failed: {e}")

                    return None  # Return None to prevent duplicate response

                if not search_results.get("success", False):
                    # Literature search failed
                    error_msg = search_results.get(
                        "error", "Unknown error in literature search"
                    )
                    print(f"❌ Literature search failed: {error_msg}")

                    status_content = (
                        f'# Research Progress\n*Researching: "{query}"*\n\n'
                    )
                    status_content += (
                        f"📚 **Step 1/5:** ✅ Query framed as research direction\n"
                    )
                    status_content += (
                        f"*Research direction:* **{framed_direction}**\n\n"
                    )
                    status_content += f"🔍 **Step 2/5:** ❌ Literature search failed\n"
                    status_content += f"*Error:* {error_msg[:200]}...\n\n"
                    status_content += "⚠️ Generating a general response instead..."
                    await update_status(status_content)

                    # Generate a fallback response
                    response = f"I've framed your query as: **{framed_direction}**\n\nHowever, I wasn't able to complete the literature search due to an error: {error_msg}\n\n"
                    response += "You might want to try:\n"
                    response += "1. Rephrasing your query to be more specific\n"
                    response += (
                        "2. Breaking your question into smaller, focused parts\n"
                    )
                    response += (
                        "3. Checking if your topic has sufficient published research"
                    )

                    # Send the response first
                    await message.channel.send(response)

                    # Safely delete the status message
                    try:
                        await status_message.delete()
                    except discord.NotFound:
                        print("Status message already deleted, skipping deletion")
                    except discord.HTTPException as http_err:
                        print(f"HTTP error when deleting status message: {http_err}")
                    except Exception as e:
                        print(f"Error deleting status message: {e}")

                    return None  # Return None to prevent duplicate response

                # Rest of the research pipeline (for successful search)
                print("📖 Literature search successful")
                status_content = f'# Research Progress\n*Researching: "{query}"*\n\n'
                status_content += (
                    f"📚 **Step 1/5:** ✅ Query framed as research direction\n"
                )
                status_content += f"*Research direction:* **{framed_direction}**\n\n"

                # Add detailed source list in Step 2
                status_content += f"🔍 **Step 2/5:** ✅ Literature search complete\n"
                citations = search_results.get("citations", [])
                status_content += f"**Sources found ({len(citations)}):**\n\n"
                # Use the helper function to format citations consistently
                status_content += self._format_citation_list(citations)
                status_content += "\n"

                status_content += (
                    f"💡 **Step 3/5:** Generating research directions...\n"
                )
                await update_status(status_content)

                # Step 3: Generate potential research directions
                try:
                    print("🧠 Step 3: Generating potential research directions...")
                    directions_result = (
                        await self.directions_generator.generate_directions(
                            search_results
                        )
                    )

                    # Check if directions generation failed
                    if not directions_result.get("success", False):
                        print(
                            f"⚠️ No research directions were generated - will proceed with literature only"
                        )
                        directions_count = 0
                        # We don't stop the pipeline here, just note there were no directions
                    else:
                        directions_count = directions_result.get("count", 0)
                        print(f"💡 Generated {directions_count} research directions")

                    status_content = (
                        f'# Research Progress\n*Researching: "{query}"*\n\n'
                    )
                    status_content += (
                        f"📚 **Step 1/5:** ✅ Query framed as research direction\n"
                    )
                    status_content += (
                        f"*Research direction:* **{framed_direction}**\n\n"
                    )

                    # Add detailed source list in Step 2
                    status_content += (
                        f"🔍 **Step 2/5:** ✅ Literature search complete\n"
                    )
                    citations = search_results.get("citations", [])
                    status_content += f"**Sources found ({len(citations)}):**\n"
                    # List up to 5 sources in the status
                    for i, citation in enumerate(citations[:5]):
                        # Truncate long citations
                        short_citation = (
                            citation[:150] + "..." if len(citation) > 150 else citation
                        )
                        status_content += f"- {short_citation}\n"
                    if len(citations) > 5:
                        status_content += (
                            f"- *(plus {len(citations) - 5} more sources)*\n"
                        )
                    status_content += "\n"

                    if directions_count > 0:
                        directions = directions_result.get("directions", [])
                        status_content += f"💡 **Step 3/5:** ✅ Generated {directions_count} research directions\n"
                        status_content += "**Research directions:**\n"
                        for i, direction in enumerate(directions):
                            title = direction.get("title", f"Direction {i+1}")
                            status_content += f"- {title}\n"
                        status_content += "\n"
                        status_content += f"⚖️ **Step 4/5:** Evaluating research directions in parallel...\n"
                    else:
                        status_content += f"💡 **Step 3/5:** ⚠️ No research directions could be generated\n\n"
                        status_content += f"⚖️ **Step 4/5:** Skipping direction evaluation (no directions)\n\n"

                    await update_status(status_content)

                    # Step 4: Review research directions
                    reviews_result = {}

                    if directions_result.get("success", False) and directions_count > 0:
                        try:
                            print("⚖️ Step 4: Evaluating research directions...")
                            reviews_result = (
                                await self.direction_reviewer.compare_directions(
                                    directions_result["directions"],
                                    search_results["results"],
                                )
                            )
                            print("✅ Direction evaluations completed")
                        except Exception as e:
                            error_msg = f"Error in direction evaluation step: {str(e)}"
                            print(f"❌ {error_msg}")

                            # We'll continue with synthesis even if evaluation fails
                            reviews_result = {"success": False, "error": error_msg}

                            status_content = (
                                f'# Research Progress\n*Researching: "{query}"*\n\n'
                            )
                            status_content += f"📚 **Step 1/5:** ✅ Query framed as research direction\n"
                            status_content += (
                                f"*Research direction:* **{framed_direction}**\n\n"
                            )
                            status_content += (
                                f"🔍 **Step 2/5:** ✅ Literature search complete\n"
                            )
                            status_content += f"*Found:* {len(search_results.get('citations', []))} relevant sources\n\n"
                            status_content += f"💡 **Step 3/5:** ✅ Generated {directions_count} research directions\n"
                            status_content += f"⚖️ **Step 4/5:** ⚠️ Direction evaluation encountered an error\n"
                            status_content += f"*Error:* {str(e)[:100]}...\n\n"
                            status_content += f"🧠 **Step 5/5:** Will continue with synthesis using available data...\n"
                            await update_status(status_content)
                    else:
                        print(
                            "⚠️ Skipping direction evaluation - no valid directions generated"
                        )

                except Exception as e:
                    error_msg = f"Error in research direction generation step: {str(e)}"
                    print(f"❌ {error_msg}")

                    # We don't want to stop the entire pipeline for this error
                    # Just mark the step as failed and continue with synthesis
                    directions_result = {
                        "success": False,
                        "error": error_msg,
                        "directions": [],
                        "count": 0,
                    }

                    status_content = (
                        f'# Research Progress\n*Researching: "{query}"*\n\n'
                    )
                    status_content += (
                        f"📚 **Step 1/5:** ✅ Query framed as research direction\n"
                    )
                    status_content += (
                        f"*Research direction:* **{framed_direction}**\n\n"
                    )

                    # Add detailed source list in Step 2
                    status_content += (
                        f"🔍 **Step 2/5:** ✅ Literature search complete\n"
                    )
                    citations = search_results.get("citations", [])
                    status_content += f"**Sources found ({len(citations)}):**\n"
                    # List up to 5 sources in the status
                    for i, citation in enumerate(citations[:5]):
                        # Truncate long citations
                        short_citation = (
                            citation[:150] + "..." if len(citation) > 150 else citation
                        )
                        status_content += f"- {short_citation}\n"
                    if len(citations) > 5:
                        status_content += (
                            f"- *(plus {len(citations) - 5} more sources)*\n"
                        )
                    status_content += "\n"

                    status_content += (
                        f"💡 **Step 3/5:** ❌ Failed to generate research directions\n"
                    )
                    status_content += f"*Error:* {str(e)[:100]}...\n\n"
                    status_content += (
                        f"⚖️ **Step 4/5:** Skipping (due to previous step failure)\n\n"
                    )
                    status_content += f"🧠 **Step 5/5:** Will attempt synthesis with literature only...\n"
                    await update_status(status_content)

                # Update status for synthesis step
                status_content = f'# Research Progress\n*Researching: "{query}"*\n\n'
                status_content += (
                    f"📚 **Step 1/5:** ✅ Query framed as research direction\n"
                )
                status_content += f"*Research direction:* **{framed_direction}**\n\n"
                status_content += f"🔍 **Step 2/5:** ✅ Literature search complete\n"
                status_content += f"*Found:* {len(search_results.get('citations', []))} relevant sources\n\n"

                # Show appropriate status for directions step
                if directions_result.get("success", False):
                    status_content += f"💡 **Step 3/5:** ✅ Generated {directions_result.get('count', 0)} research directions\n"
                    if directions_result.get("directions", []):
                        direction_sample = directions_result.get("directions", [])[
                            0
                        ].get("title", "Untitled")
                        status_content += f"*Sample direction:* {direction_sample}\n\n"
                else:
                    status_content += (
                        f"💡 **Step 3/5:** ⚠️ No research directions generated\n\n"
                    )

                # Show appropriate status for evaluation step
                if reviews_result.get("success", False):
                    status_content += f"⚖️ **Step 4/5:** ✅ Research directions evaluated in parallel\n"
                    # Add a brief summary of the evaluation if available
                    if reviews_result.get("comparison", ""):
                        comparison_text = reviews_result.get("comparison", "")

                        # Remove hashtags and clean up formatting
                        comparison_text = re.sub(
                            r"###\s*\d+\.\s*STRATEGIC RANKING:",
                            "Strategic ranking:",
                            comparison_text,
                        )
                        comparison_text = re.sub(
                            r"###\s*", "", comparison_text
                        )  # Remove other hashtags
                        comparison_text = re.sub(
                            r"#\s*", "", comparison_text
                        )  # Remove single hashtags

                        # Get the first 200 chars for the summary
                        comparison_summary = comparison_text[:200]
                        if len(comparison_summary) > 0:
                            status_content += (
                                f"*Evaluation summary:* {comparison_summary}...\n"
                            )
                    status_content += "\n"
                else:
                    status_content += (
                        f"⚖️ **Step 4/5:** ⚠️ Direction evaluation skipped or failed\n\n"
                    )

                status_content += (
                    f"🧠 **Step 5/5:** Synthesizing final recommendations...\n"
                )
                await update_status(status_content)

                # Step 5: Synthesize final recommendations
                try:
                    print("🧠 Step 5: Synthesizing final recommendations...")
                    synthesis_result = await self.recommendation_synthesizer.synthesize_recommendations(
                        query, search_results, directions_result, reviews_result
                    )
                    print("✅ Recommendation synthesis completed")

                    # Final status update for successful synthesis
                    status_content = (
                        f'# Research Progress\n*Researching: "{query}"*\n\n'
                    )
                    status_content += (
                        f"📚 **Step 1/5:** ✅ Query framed as research direction\n"
                    )
                    status_content += (
                        f"*Research direction:* **{framed_direction}**\n\n"
                    )

                    # Add detailed source list in Step 2
                    status_content += (
                        f"🔍 **Step 2/5:** ✅ Literature search complete\n"
                    )
                    citations = search_results.get("citations", [])
                    status_content += f"**Sources found ({len(citations)}):**\n\n"
                    # Use the helper function to format citations consistently
                    status_content += self._format_citation_list(citations)
                    status_content += "\n"

                    # List all research directions in Step 3
                    if (
                        directions_result.get("success", False)
                        and directions_result.get("count", 0) > 0
                    ):
                        directions = directions_result.get("directions", [])
                        status_content += f"💡 **Step 3/5:** ✅ Generated {len(directions)} research directions\n"
                        status_content += "**Research directions:**\n"
                        for i, direction in enumerate(directions):
                            title = direction.get("title", f"Direction {i+1}")
                            status_content += f"- {title}\n"
                        status_content += "\n"
                    else:
                        status_content += (
                            f"💡 **Step 3/5:** ⚠️ No research directions generated\n\n"
                        )

                    # Show appropriate status for evaluation step
                    if reviews_result.get("success", False):
                        status_content += (
                            f"⚖️ **Step 4/5:** ✅ Research directions evaluated\n"
                        )
                        # Add a brief summary of the evaluation if available
                        if reviews_result.get("comparison", ""):
                            comparison_text = reviews_result.get("comparison", "")

                            # Remove hashtags and clean up formatting
                            comparison_text = re.sub(
                                r"###\s*\d+\.\s*STRATEGIC RANKING:",
                                "Strategic ranking:",
                                comparison_text,
                            )
                            comparison_text = re.sub(
                                r"###\s*", "", comparison_text
                            )  # Remove other hashtags
                            comparison_text = re.sub(
                                r"#\s*", "", comparison_text
                            )  # Remove single hashtags

                            # Get the first 200 chars for the summary
                            comparison_summary = comparison_text[:200]
                            if len(comparison_summary) > 0:
                                status_content += (
                                    f"*Evaluation summary:* {comparison_summary}...\n"
                                )
                        status_content += "\n"
                    else:
                        status_content += f"⚖️ **Step 4/5:** ⚠️ Direction evaluation skipped or failed\n\n"

                    status_content += f"🧠 **Step 5/5:** ✅ Synthesis complete\n\n"
                    status_content += (
                        f"Research completed! *Detailed results provided below.*"
                    )
                    await update_status(status_content)

                except Exception as e:
                    error_msg = f"Error in synthesis step: {str(e)}"
                    print(f"❌ {error_msg}")

                    # Update status to show synthesis failure
                    status_content = (
                        f'# Research Progress\n*Researching: "{query}"*\n\n'
                    )
                    status_content += (
                        f"📚 **Step 1/5:** ✅ Query framed as research direction\n"
                    )
                    status_content += (
                        f"*Research direction:* **{framed_direction}**\n\n"
                    )

                    # Add detailed source list in Step 2
                    status_content += (
                        f"🔍 **Step 2/5:** ✅ Literature search complete\n"
                    )
                    citations = search_results.get("citations", [])
                    status_content += f"**Sources found ({len(citations)}):**\n\n"
                    # Use the helper function to format citations consistently
                    status_content += self._format_citation_list(citations)
                    status_content += "\n"

                    # List all research directions in Step 3
                    if (
                        directions_result.get("success", False)
                        and directions_result.get("count", 0) > 0
                    ):
                        directions = directions_result.get("directions", [])
                        status_content += f"💡 **Step 3/5:** ✅ Generated {len(directions)} research directions\n"
                        status_content += "**Research directions:**\n"
                        for i, direction in enumerate(directions):
                            title = direction.get("title", f"Direction {i+1}")
                            status_content += f"- {title}\n"
                        status_content += "\n"
                    else:
                        status_content += (
                            f"💡 **Step 3/5:** ⚠️ No research directions generated\n\n"
                        )

                    # Show appropriate status for evaluation step
                    if reviews_result.get("success", False):
                        status_content += (
                            f"⚖️ **Step 4/5:** ✅ Research directions evaluated\n"
                        )
                        # Add a brief summary of the evaluation if available
                        if reviews_result.get("comparison", ""):
                            comparison_text = reviews_result.get("comparison", "")

                            # Remove hashtags and clean up formatting
                            comparison_text = re.sub(
                                r"###\s*\d+\.\s*STRATEGIC RANKING:",
                                "Strategic ranking:",
                                comparison_text,
                            )
                            comparison_text = re.sub(
                                r"###\s*", "", comparison_text
                            )  # Remove other hashtags
                            comparison_text = re.sub(
                                r"#\s*", "", comparison_text
                            )  # Remove single hashtags

                            # Get the first 200 chars for the summary
                            comparison_summary = comparison_text[:200]
                            if len(comparison_summary) > 0:
                                status_content += (
                                    f"*Evaluation summary:* {comparison_summary}...\n"
                                )
                        status_content += "\n"
                    else:
                        status_content += f"⚖️ **Step 4/5:** ⚠️ Direction evaluation skipped or failed\n\n"

                    status_content += f"🧠 **Step 5/5:** ❌ Synthesis step failed\n"
                    status_content += f"*Error:* {str(e)[:100]}...\n\n"
                    status_content += f"**Research incomplete.** *Generating a basic response with available information...*"
                    await update_status(status_content)

                    # Create a fallback synthesis result
                    synthesis_result = {
                        "success": False,
                        "error": error_msg,
                        "recommendations": f"I couldn't synthesize a detailed response due to an error in processing: {str(e)}",
                        "next_steps": [
                            {
                                "number": "1",
                                "description": "Try refining your query to be more specific",
                            },
                            {
                                "number": "2",
                                "description": "Break down your research question into smaller parts",
                            },
                            {
                                "number": "3",
                                "description": "Try again with a different formulation",
                            },
                        ],
                        "citations": search_results.get("citations", []),
                    }

                # Create the final response without markdown artifacts
                response = "Research on: " + framed_direction + "\n\n"

                if synthesis_result.get("success", False):
                    response += (
                        "Recommendations\n"
                        + synthesis_result.get(
                            "recommendations", "No specific recommendations available."
                        )
                        + "\n\n"
                    )

                    # Always add next steps
                    if synthesis_result.get("next_steps", []):
                        response += "Next Steps\n"
                        for step in synthesis_result.get("next_steps", []):
                            response += f"{step.get('number', '•')}. {step.get('description', '')}\n"
                        response += "\n"

                    # Always add citations with better formatting
                    citations = synthesis_result.get("citations", [])
                    if citations:
                        response += "Relevant Papers\nThese papers are directly relevant to your research question:\n\n"
                        response += self._format_citation_list(
                            citations, max_display=8
                        )  # Show more papers in the final response
                        response += "\n"
                else:
                    # Fallback if synthesis failed
                    response += "I was able to find information on your query, but had trouble synthesizing recommendations.\n\n"
                    response += "Here's a summary of what I found:\n\n"

                    # Include a preview of search results
                    results_preview = (
                        search_results.get("results", "")[:500] + "..."
                        if len(search_results.get("results", "")) > 500
                        else search_results.get("results", "")
                    )
                    response += f"{results_preview}\n\n"

                    # Add citations
                    citations = search_results.get("citations", [])
                    if citations:
                        response += "Relevant Papers\nThese papers are relevant to your research question:\n\n"
                        response += self._format_citation_list(citations, max_display=8)
                        response += "\n"

                    # Add basic next steps
                    response += "Suggested Next Steps\n"
                    response += "1. Review the literature summary and papers above\n"
                    response += "2. Consider narrowing your research focus\n"
                    response += "3. Try a follow-up query with 'refine:' prefix\n"

                # Check if response exceeds Discord's character limit
                if len(response) <= 2000:
                    # Send the single response
                    sent_response = await message.channel.send(response)

                    # Keep the status message visible - don't delete it
                    # We'll return None to prevent sending duplicate messages
                    return None
                else:
                    # Split response into chunks of 2000 characters or less,
                    # ensuring we split at line breaks to preserve Markdown formatting
                    chunks = []
                    lines = response.split("\n")

                    current_chunk = ""
                    for line in lines:
                        # If adding this line would exceed the limit
                        if (
                            len(current_chunk) + len(line) + 1 > 1950
                        ):  # +1 for the newline, 1950 to leave room for thread numbering
                            # Save current chunk and start a new one
                            if current_chunk:
                                chunks.append(current_chunk)
                                current_chunk = line
                            else:
                                # If a single line is too long (rare case), split it by words
                                words = line.split(" ")
                                line_chunk = ""
                                for word in words:
                                    if len(line_chunk) + len(word) + 1 <= 1950:
                                        if line_chunk:
                                            line_chunk += " "
                                        line_chunk += word
                                    else:
                                        chunks.append(line_chunk)
                                        line_chunk = word

                                if line_chunk:
                                    current_chunk = line_chunk
                        else:
                            # Add the line to the current chunk
                            if current_chunk:
                                current_chunk += "\n"
                            current_chunk += line

                    # Add the last chunk if it exists
                    if current_chunk:
                        chunks.append(current_chunk)

                    # No thread numbering, keep the chunks as is
                    total_chunks = len(chunks)

                    # Send all chunks in order
                    for chunk in chunks:
                        await message.channel.send(chunk)

                    # Keep the status message visible - don't delete it
                    # We'll return None to prevent sending duplicate messages
                    return None

            except Exception as e:
                # Handle errors in the research pipeline
                error_message = (
                    f"An error occurred during the research process: {str(e)}"
                )
                print(f"❌ Error in research pipeline: {str(e)}")
                import traceback

                traceback.print_exc()

                try:
                    # Update the status message with the error
                    error_content = f"❌ **Research Error**\n\nAn error occurred while processing your request: {str(e)[:200]}...\n\nPlease try again with a different query."
                    await update_status(error_content)

                    # Wait a moment before sending the detailed error response
                    await asyncio.sleep(1.0)

                    # Send the error response as a new message
                    response = f"I encountered an error while researching your query: {str(e)}\n\nPlease try again with a different formulation."
                    await message.channel.send(response)

                    # Safely delete the status message
                    try:
                        await status_message.delete()
                    except discord.NotFound:
                        print("Status message already deleted, skipping deletion")
                    except discord.HTTPException as http_err:
                        print(f"HTTP error when deleting status message: {http_err}")
                    except Exception as e:
                        print(f"Error deleting status message: {e}")
                except Exception as inner_e:
                    print(f"Error during error handling: {inner_e}")

                return None  # Return None since we already sent the response

        except Exception as e:
            error_message = f"An error occurred while processing your request: {str(e)}"
            print(f"❌ Error in run method: {str(e)}")
            import traceback

            traceback.print_exc()

            # Send the error directly rather than returning it
            # This ensures consistent message handling pattern throughout the code
            await message.channel.send(error_message)
            return None  # Return None since we've sent the message directly
