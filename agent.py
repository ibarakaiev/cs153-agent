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
                model=self.model,
                messages=messages,
                **kwargs
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
        if not self.api_key:
            print("❌ Perplexity API key is not set")
            return {
                "success": False,
                "error": "Perplexity API key not configured",
                "results": "",
                "citations": []
            }
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
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
                {"role": "system", "content": "You are a research assistant conducting literature reviews. Be thorough but concise. Prioritize academic sources and provide proper citations."},
                {"role": "user", "content": search_prompt}
            ],
            "temperature": 0.2  # Lower temperature for more factual responses
            # No max_tokens - let it generate a complete response
        }
        
        try:
            print(f"🔍 Sending search request to Perplexity API for: {query[:50]}...")
            print(f"📝 Using model: {data['model']}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    response_text = await response.text()
                    print(f"📡 Perplexity API response status: {response.status}")
                    
                    if response.status != 200:
                        print(f"❌ Error response: {response_text[:200]}...")
                        return {
                            "success": False,
                            "error": f"Search failed with status {response.status}: {response_text[:500]}",
                            "results": "",
                            "citations": []
                        }
                        
                    try:
                        response_json = json.loads(response_text)
                        print(f"📊 Response JSON keys: {list(response_json.keys())}")
                        
                        # Extract content from the correct location in the response
                        # The Perplexity API returns the content in choices[0].message.content
                        if "choices" in response_json and len(response_json["choices"]) > 0:
                            content = response_json["choices"][0]["message"]["content"]
                            print(f"✅ Successfully extracted content: {len(content)} chars")
                            
                            # Extract citations from the message content using regex
                            citations = []
                            citation_matches = re.findall(r'\[(\d+)\]\s+(.*?)(?=\n\[|\n\n|$)', content, re.DOTALL)
                            for match in citation_matches:
                                citations.append(match[1].strip())
                                
                            print(f"📚 Extracted {len(citations)} citations from content")
                            
                            return {
                                "success": True,
                                "results": content,
                                "citations": citations,
                                "query": query
                            }
                        else:
                            print("❌ No 'choices' found in response")
                            print(f"📄 Response: {response_text[:200]}...")
                            return {
                                "success": False,
                                "error": "Invalid response format from Perplexity API",
                                "results": "",
                                "citations": []
                            }
                    except json.JSONDecodeError:
                        print("❌ Failed to parse JSON response")
                        return {
                            "success": False,
                            "error": f"Failed to parse JSON response: {response_text[:200]}...",
                            "results": "",
                            "citations": []
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
                "citations": []
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
                    {"role": "system", "content": "You are a research question formulation assistant. Your job is to convert informal queries into precise, searchable research directions."},
                    {"role": "user", "content": framing_prompt}
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
                "output_tokens": response.get("output_tokens", 0)
            }
        except Exception as e:
            error_message = f"Error framing research direction: {str(e)}"
            print(f"❌ {error_message}")
            return {
                "success": False,
                "error": error_message,
                "research_direction": query,  # Fall back to original query
                "original_query": query
            }


class DirectionReviewer:
    """Reviews and evaluates potential research directions"""
    
    def __init__(self, llm_provider: MistralProvider):
        self.llm_provider = llm_provider
    
    async def review_direction(self, direction: Dict[str, str], literature_context: str) -> Dict[str, Any]:
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
                {"role": "system", "content": """You are a research direction evaluator specializing in:
1. Validating research gaps
2. Assessing technical feasibility
3. Evaluating potential impact
4. Resource planning
5. Competitive analysis

Provide concrete, specific assessments backed by the literature context."""},
                {"role": "user", "content": review_prompt}
            ]
        )
        
        # Extract sections using regex
        review_content = response["content"]
        review = {
            "title": direction.get('title', 'Untitled'),
            "question": direction.get('question', 'No question provided'),
            "feasibility": self._extract_section(review_content, "TECHNICAL FEASIBILITY"),
            "novelty": self._extract_section(review_content, "GAP VALIDATION"),
            "impact": self._extract_section(review_content, "RESEARCH IMPACT"),
            "pros": self._extract_section(review_content, "UNIQUE ADVANTAGES"),
            "cons": self._extract_section(review_content, "POTENTIAL ROADBLOCKS"),
            "resources": self._extract_section(review_content, "RESOURCE REQUIREMENTS"),
            "priority": self._extract_section(review_content, "FINAL VERDICT"),
        }
        
        return review
    
    async def compare_directions(self, directions: List[Dict[str, str]], literature_context: str) -> Dict[str, Any]:
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
                "reviews": []
            }
        
        print(f"🔍 Reviewing {len(directions)} research directions in parallel...")
        
        # Process each direction review in parallel
        review_tasks = [self.review_direction(direction, literature_context) for direction in directions]
        reviews = await asyncio.gather(*review_tasks)
        
        print("✅ Direction reviews completed")
        
        # Generate a comparison of the directions
        comparison = await self._generate_comparison(directions, reviews)
        
        return {
            "success": True,
            "reviews": reviews,
            "comparison": comparison,
            "count": len(reviews)
        }
    
    async def _generate_comparison(self, directions: List[Dict[str, str]], reviews: List[Dict[str, str]]) -> str:
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
                {"role": "system", "content": "You are a research advisor who compares different research directions and provides strategic recommendations."},
                {"role": "user", "content": comparison_prompt}
            ],
            max_tokens=1000
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
        section_pattern = section_name.replace('_', '[_ ]?')
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

    def _format_reviews_summary(self, reviews: List[Dict[str, str]]) -> str:
        """Format reviews into a concise summary for comparison"""
        summary = ""
        
        for i, review in enumerate(reviews):
            title = review.get("title", f"Direction {i+1}")
            question = review.get("question", "No question provided")
            priority = review.get("priority", "No priority assessment")
            
            summary += f"DIRECTION {i+1}: {title}\n"
            summary += f"QUESTION: {question}\n"
            summary += f"PRIORITY: {priority}\n"
            
            # Add key metrics if available
            for key in ["feasibility", "novelty", "impact"]:
                if key in review and review[key]:
                    # Truncate long values
                    value = review[key]
                    if len(value) > 100:
                        value = value[:97] + "..."
                    summary += f"{key.upper()}: {value}\n"
            
            summary += "\n"
            
        return summary


class ResearchDirectionsGenerator:
    """Generates potential research directions based on literature review results"""
    
    def __init__(self, llm_provider: MistralProvider):
        self.llm_provider = llm_provider
        
    async def generate_directions(self, search_results: Dict[str, Any], count: int = 3) -> Dict[str, Any]:
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
                "directions": []
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
                "directions": []
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
                    {"role": "system", "content": """You are a research direction generator who identifies promising research opportunities based on literature reviews.

You MUST follow the output format instructions precisely, using the exact section headers requested with the exact capitalization and underscores shown."""},
                    {"role": "user", "content": generation_prompt}
                ]
            )
            
            # Parse the response to extract research directions
            generated_content = response["content"]
            print(f"✅ Received {len(generated_content)} chars of generated content")
            
            # Debug: show a preview of the generated content
            preview = generated_content[:300].replace('\n', ' ')
            print(f"📄 Content preview: {preview}...")
            
            directions = []
            
            # Extract each direction using regex
            for i in range(1, count + 1):
                direction = {}
                
                # Extract title
                title_pattern = f"DIRECTION_{i}_TITLE:(.*?)(?:DIRECTION_{i}_QUESTION:|$)"
                title_match = re.search(title_pattern, generated_content, re.DOTALL)
                if title_match:
                    direction["title"] = title_match.group(1).strip()
                    print(f"✅ Found direction {i} title: {direction['title']}")
                else:
                    print(f"❓ Missing title for direction {i}")
                
                # Extract question (only if title was found)
                if "title" in direction:
                    question_pattern = f"DIRECTION_{i}_QUESTION:(.*?)(?:DIRECTION_{i}_RATIONALE:|$)"
                    question_match = re.search(question_pattern, generated_content, re.DOTALL)
                    if question_match:
                        direction["question"] = question_match.group(1).strip()
                    
                    # Extract rationale
                    rationale_pattern = f"DIRECTION_{i}_RATIONALE:(.*?)(?:DIRECTION_{i}_METHODOLOGY:|$)"
                    rationale_match = re.search(rationale_pattern, generated_content, re.DOTALL)
                    if rationale_match:
                        direction["rationale"] = rationale_match.group(1).strip()
                    
                    # Extract methodology
                    methodology_pattern = f"DIRECTION_{i}_METHODOLOGY:(.*?)(?:DIRECTION_{i}_GAPS:|DIRECTION_{i+1}_TITLE:|$)"
                    methodology_match = re.search(methodology_pattern, generated_content, re.DOTALL)
                    if methodology_match:
                        direction["methodology"] = methodology_match.group(1).strip()
                    
                    # Extract gaps
                    gaps_pattern = f"DIRECTION_{i}_GAPS:(.*?)(?:DIRECTION_{i+1}_TITLE:|$)"
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
                        if current_direction.get("title") and current_direction.get("question"):
                            directions.append(current_direction)
                            print(f"✅ Added direction from fallback: {current_direction['title']}")
                        
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
                    print(f"✅ Added final direction from fallback: {current_direction['title']}")
            
            print(f"💡 Final count: {len(directions)} research directions")
            
            return {
                "success": len(directions) > 0,
                "directions": directions,
                "count": len(directions),
                "input_tokens": response.get("input_tokens", 0),
                "output_tokens": response.get("output_tokens", 0)
            }
        except Exception as e:
            error_message = f"Error generating research directions: {str(e)}"
            print(f"❌ {error_message}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": error_message,
                "directions": []
            }


class RecommendationSynthesizer:
    """Synthesizes final research recommendations and next steps"""
    
    def __init__(self, llm_provider: MistralProvider):
        self.llm_provider = llm_provider
        
    def _build_full_synthesis_prompt(self, query: str, lit_review: str, comparison: str) -> str:
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
    
    def _build_directions_only_prompt(self, query: str, lit_review: str, directions: List[Dict[str, str]]) -> str:
        """Build synthesis prompt when we have literature review and directions but no reviews"""
        directions_text = "\n\n".join([
            f"DIRECTION {i+1}: {direction.get('title', 'Untitled')}\n{direction.get('question', 'No question provided')}"
            for i, direction in enumerate(directions[:3])
        ])
        
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
    
    async def synthesize_recommendations(self, query: str, search_results: Dict[str, Any], 
                                         directions_result: Dict[str, Any], 
                                         reviews_result: Dict[str, Any]) -> Dict[str, Any]:
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
                    "next_steps": []
                }
            
            # Extract the necessary information from inputs
            lit_review = search_results.get("results", "")[:2000]  # Limit to avoid token overflow
            
            # Build different prompts based on available data
            if reviews_result.get("success", False) and reviews_result.get("comparison", ""):
                # We have both directions and reviews
                comparison = reviews_result.get("comparison", "")
                prompt = self._build_full_synthesis_prompt(query, lit_review, comparison)
            elif directions_result.get("success", False) and directions_result.get("directions", []):
                # We have directions but no reviews
                directions = directions_result.get("directions", [])
                prompt = self._build_directions_only_prompt(query, lit_review, directions)
            else:
                # We only have literature review
                prompt = self._build_literature_only_prompt(query, lit_review)
            
            # Get synthesis from LLM with stronger formatting instructions
            response = await self.llm_provider.generate_completion_sync(
                [
                    {"role": "system", "content": """You are a research advisor who synthesizes research findings and provides actionable recommendations.

You MUST follow the output format instructions precisely. Your response should ALWAYS include the exact section headers requested (RECOMMENDATIONS:, NEXT_STEPS:, etc.) with no variations. 

Format next steps as numbered lists (1., 2., etc.) and ensure each section has substantive content."""},
                    {"role": "user", "content": prompt}
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
                    "next_steps": [{"number": "1", "description": "Try a more specific research query"}]
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
                for section_name in ["SYNTHESIS", "SUMMARY", "OVERVIEW", "ANALYSIS", "RESULTS"]:
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
                    {"number": "1", "description": "Conduct a more focused literature review on the specific aspects identified above"}, 
                    {"number": "2", "description": "Consider reaching out to experts in the field for additional insights"}
                ]
            
            print(f"✅ Successfully synthesized recommendations ({len(recommendations)} chars) and {len(next_steps)} next steps")
            
            return {
                "success": True,
                "recommendations": recommendations,
                "next_steps": next_steps,
                "full_synthesis": content
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
                "next_steps": [{"number": "1", "description": "Try a more specific research query"}]
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
        section_pattern = section_name.replace('_', '[_ ]?')
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
                {"number": "1", "description": "Review the literature findings and select a specific focus area"},
                {"number": "2", "description": "Develop a detailed research plan based on the recommendations"}
            ]
        
        # Look for numbered steps (1., 2., etc.)
        step_matches = re.findall(r'(\d+[\.\)\-])\s+(.*?)(?=\n\s*\d+[\.\)\-]|\n\n|$)', steps_text, re.DOTALL)
        
        if step_matches:
            for match in step_matches:
                number, text = match
                steps.append({
                    "number": number.strip('.)-'),
                    "description": text.strip()
                })
        else:
            # Alternative parsing for bullet points or other formats
            lines = steps_text.split('\n')
            current_step = None
            step_number = 1
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                
                # Check if this looks like a new step
                if stripped.startswith('•') or stripped.startswith('-') or stripped.startswith('*') or (len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ['.', ')', ':', '-']):
                    if current_step:
                        steps.append(current_step)
                    current_step = {
                        "number": str(step_number),
                        "description": stripped.lstrip('•-*123456789.) :')
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
            parts = re.split(r'(?<=[.!?])\s+', steps_text)
            for i, part in enumerate(parts[:3]):  # Limit to 3 steps
                if len(part.strip()) > 10:  # Only add if meaningful content
                    steps.append({
                        "number": str(i+1),
                        "description": part.strip()
                    })
        
        # If still empty, add a default step
        if not steps:
            steps.append({
                "number": "1",
                "description": "Review the literature and synthesize key findings into a research plan"
            })
        
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
                print(f"📄 Search results preview: {search_results.get('results', '')[:200]}...")
                
                # Step 3: Generate potential research directions
                print("🧠 Step 3: Generating potential research directions...")
                directions_result = await self.directions_generator.generate_directions(search_results)
                print(f"💡 Generated {directions_result.get('count', 0)} research directions")
                
                # Step 4: Review research directions
                reviews_result = {}
                if directions_result.get("success", False) and directions_result.get("count", 0) > 0:
                    print("⚖️ Step 4: Evaluating research directions...")
                    reviews_result = await self.direction_reviewer.compare_directions(
                        directions_result["directions"], 
                        search_results["results"]
                    )
                    print("✅ Direction evaluations completed")
                    
                    # Send a message with the evaluation summary
                    if reviews_result.get("success", False) and reviews_result.get("comparison"):
                        evaluation_summary = f"## Direction Evaluation Summary\n{reviews_result['comparison']}"
                        # Check if summary exceeds Discord's character limit
                        if len(evaluation_summary) > 2000:
                            evaluation_summary = evaluation_summary[:1997] + "..."
                        await message.channel.send(evaluation_summary)
                else:
                    print("⚠️ Skipping direction evaluation - no valid directions generated")
                
                # Step 5: Synthesize final recommendations
                print("🧠 Step 5: Synthesizing final recommendations...")
                synthesis_result = await self.recommendation_synthesizer.synthesize_recommendations(
                    query,
                    search_results,
                    directions_result,
                    reviews_result
                )
                print("✅ Recommendation synthesis completed")
                
                return {
                    "original_query": query,
                    "framed_direction": framed_direction,
                    "search_results": search_results,
                    "directions_result": directions_result,
                    "reviews_result": reviews_result,
                    "synthesis_result": synthesis_result,
                    "success": True
                }
            else:
                print(f"❌ Literature search failed: {search_results.get('error', 'Unknown error')}")
                return {
                    "original_query": query,
                    "framed_direction": framed_direction,
                    "search_results": search_results,
                    "success": False
                }
        except Exception as e:
            error_message = f"Error in research query processing: {str(e)}"
            print(f"❌ {error_message}")
            import traceback
            traceback.print_exc()
            return {
                "original_query": query,
                "success": False,
                "error": error_message
            }

    async def run(self, message: discord.Message):
        try:
            query = message.content
            print(f"🤖 Starting to process query: {query}")
            
            # Check if this is a refinement of a previous query
            is_refinement = False
            original_query = query
            
            # Check if message starts with "refine:" or similar refinement indicators
            refinement_prefixes = ["refine:", "iterate:", "narrow:", "focus:"]
            for prefix in refinement_prefixes:
                if query.lower().startswith(prefix):
                    is_refinement = True
                    query = query[len(prefix):].strip()
                    print(f"📌 Detected refinement request: {query}")
                    break
            
            # Message the user that processing has started
            status_text = "🧠 Starting research process..."
            if is_refinement:
                status_text = "🔍 Starting to refine your research direction..."
            
            status_message = await message.channel.send(status_text)
            
            try:
                # Step 1: Frame the query as a research direction
                await status_message.edit(content=f"📚 **Step 1/5:** Framing your query as a research direction...\n\n*Original query:* {query}")
                print("📚 Step 1: Framing research direction...")
                framed = await self.research_framer.frame_as_research_direction(query)
                framed_direction = framed["research_direction"]
                print(f"🎯 Framed Direction: {framed_direction}")
                
                # Update status message with framed direction
                await status_message.edit(content=f"📚 **Step 1/5:** Query framed as research direction ✅\n\n*Research direction:* **{framed_direction}**\n\n🔍 **Step 2/5:** Searching scientific literature...")
                
                # Step 2: Perform literature search
                print("🔍 Step 2: Conducting literature search...")
                search_results = await self.search_provider.search(framed_direction)
                
                if not search_results.get("success", False):
                    # Literature search failed
                    error_msg = search_results.get("error", "Unknown error in literature search")
                    print(f"❌ Literature search failed: {error_msg}")
                    
                    await status_message.edit(content=f"📚 **Step 1/5:** Query framed as research direction ✅\n\n*Research direction:* **{framed_direction}**\n\n🔍 **Step 2/5:** ❌ Literature search failed\n\n*Error:* {error_msg[:200]}...\n\n⚠️ Generating a general response instead...")
                    
                    # Generate a fallback response
                    response = f"I've framed your query as: **{framed_direction}**\n\nHowever, I wasn't able to complete the literature search due to an error: {error_msg}\n\n"
                    response += "You might want to try:\n"
                    response += "1. Rephrasing your query to be more specific\n"
                    response += "2. Breaking your question into smaller, focused parts\n"
                    response += "3. Checking if your topic has sufficient published research"
                    
                    # Delete status message and return response
                    await status_message.delete()
                    return response
                
                # Rest of the research pipeline (for successful search)
                print("📖 Literature search successful")
                await status_message.edit(content=f"📚 **Step 1/5:** Query framed as research direction ✅\n\n*Research direction:* **{framed_direction}**\n\n🔍 **Step 2/5:** Literature search complete ✅\n\n💡 **Step 3/5:** Generating research directions...")
                
                # Step 3: Generate potential research directions
                print("🧠 Step 3: Generating potential research directions...")
                directions_result = await self.directions_generator.generate_directions(search_results)
                print(f"💡 Generated {directions_result.get('count', 0)} research directions")
                
                await status_message.edit(content=f"📚 **Step 1/5:** Query framed as research direction ✅\n\n*Research direction:* **{framed_direction}**\n\n🔍 **Step 2/5:** Literature search complete ✅\n\n💡 **Step 3/5:** Generated {directions_result.get('count', 0)} research directions ✅\n\n⚖️ **Step 4/5:** Evaluating research directions...")
                
                # Step 4: Review research directions
                reviews_result = {}
                if directions_result.get("success", False) and directions_result.get("count", 0) > 0:
                    print("⚖️ Step 4: Evaluating research directions...")
                    reviews_result = await self.direction_reviewer.compare_directions(
                        directions_result["directions"], 
                        search_results["results"]
                    )
                    print("✅ Direction evaluations completed")
                    
                    # Send a message with the evaluation summary
                    if reviews_result.get("success", False) and reviews_result.get("comparison"):
                        evaluation_summary = f"## Direction Evaluation Summary\n\n{reviews_result['comparison']}"
                        # Check if summary exceeds Discord's character limit
                        if len(evaluation_summary) > 2000:
                            evaluation_summary = evaluation_summary[:1997] + "..."
                        await message.channel.send(evaluation_summary)
                else:
                    print("⚠️ Skipping direction evaluation - no valid directions generated")
                
                await status_message.edit(content=f"📚 **Step 1/5:** Query framed as research direction ✅\n\n*Research direction:* **{framed_direction}**\n\n🔍 **Step 2/5:** Literature search complete ✅\n\n💡 **Step 3/5:** Generated {directions_result.get('count', 0)} research directions ✅\n\n⚖️ **Step 4/5:** Research directions evaluated ✅\n\n🧠 **Step 5/5:** Synthesizing final recommendations...")
                
                # Step 5: Synthesize final recommendations
                print("🧠 Step 5: Synthesizing final recommendations...")
                synthesis_result = await self.recommendation_synthesizer.synthesize_recommendations(
                    query,
                    search_results,
                    directions_result,
                    reviews_result
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
                else:
                    # Fallback if synthesis failed
                    response += f"I was able to find information on your query, but had trouble synthesizing recommendations.\n\n"
                    response += f"Here's a summary of what I found:\n\n"
                    
                    # Include a preview of search results
                    results_preview = search_results.get("results", "")[:500] + "..." if len(search_results.get("results", "")) > 500 else search_results.get("results", "")
                    response += f"{results_preview}\n\n"
                    
                    # Add basic next steps
                    response += "### Suggested Next Steps\n"
                    response += "1. Review the literature summary above\n"
                    response += "2. Consider narrowing your research focus\n"
                    response += "3. Try a follow-up query with 'refine:' prefix\n"
                
                # Delete status message
                await status_message.delete()

                # Check if response exceeds Discord's character limit
                if len(response) <= 2000:
                    return response
                else:
                    # Split response into chunks of 2000 characters or less
                    # We'll return the first chunk and send the rest as follow-up messages
                    chunks = []
                    current_chunk = ""
                    
                    # Try to split at paragraph breaks to make the chunks more readable
                    paragraphs = response.split("\n\n")
                    for paragraph in paragraphs:
                        if len(current_chunk) + len(paragraph) + 2 <= 2000:  # +2 for the "\n\n"
                            if current_chunk:
                                current_chunk += "\n\n"
                            current_chunk += paragraph
                        else:
                            # If adding this paragraph would exceed the limit
                            if current_chunk:
                                chunks.append(current_chunk)
                                current_chunk = paragraph
                            else:
                                # If a single paragraph is too long, split it by characters
                                if len(paragraph) > 2000:
                                    for i in range(0, len(paragraph), 1900):
                                        chunks.append(paragraph[i:i+1900])
                                else:
                                    chunks.append(paragraph)
                    
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # Send follow-up chunks to the channel
                    if len(chunks) > 1:
                        for i in range(1, len(chunks)):
                            await message.channel.send(chunks[i])
                            
                    # Return the first chunk to be sent as the reply
                    return chunks[0]
                
            except Exception as e:
                # Handle errors in the research pipeline
                error_message = f"An error occurred during the research process: {str(e)}"
                print(f"❌ Error in research pipeline: {str(e)}")
                import traceback
                traceback.print_exc()
                
                await status_message.edit(content=f"❌ **Research Error**\n\nAn error occurred while processing your request: {str(e)[:200]}...\n\nPlease try again with a different query.")
                
                response = f"I encountered an error while researching your query: {str(e)}\n\nPlease try again with a different formulation."
                
                # Delete status message and return response
                await status_message.delete()
                return response
            
        except Exception as e:
            error_message = f"An error occurred while processing your request: {str(e)}"
            print(f"❌ Error in run method: {str(e)}")
            import traceback
            traceback.print_exc()
            return error_message  # Important: Return the error message to avoid empty response
