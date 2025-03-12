- [x] **Task 1: Read user input**

  - [x] **Subtask 1.1**: Read user's discord message
  - [x] **Subtask 1.2**: Frame as a broad research direction to pass into next step

- [ ] **Task 2: Implement Literature Review Search with Perplexity Sonar Pro**

  - [x] **Subtask 2.1**: Set up `search` Mistral function to query Perplexity Sonar Pro
  - [x] **Subtask 2.2**: Configure query parameters and parsing of search results
  - [x] **Subtask 2.3**: Test the integration to ensure relevant literature is retrieved

- [ ] **Task 3: Create Function to Parse Search Results**

  - [x] **Subtask 3.1**: Develop `parse_search_results` Mistral function
  - [x] **Subtask 3.2**: Extract key metadata (methodologies, findings, references)

- [ ] **Task 4: Generate Potential Research Directions**

  - [x] **Subtask 4.1**: Develop `generate_research_directions` Mistral function
  - [x] **Subtask 4.2**: Use parsed data to propose multiple hypotheses or directions
  - [x] **Subtask 4.3**: Ensure each direction is linked to relevant literature references

- [ ] **Task 5: Conduct Parallel Review of Directions**

  - [ ] **Subtask 5.1**: Build `review_research_directions` Mistral function to evaluate feasibility, novelty, and alignment
  - [ ] **Subtask 5.2**: Compare directions in parallel for pros/cons, resources needed, potential impact
  - [ ] **Subtask 5.3**: Summarize each review's findings in a standardized format

- [ ] **Task 6: Synthesize Final Recommendations**

  - [ ] **Subtask 6.1**: Create `synthesize_next_steps` Mistral function
  - [ ] **Subtask 6.2**: Consolidate parallel reviews into ranked lists or summaries
  - [ ] **Subtask 6.3**: Provide clear next steps or suggested experimental approaches

- [ ] **Task 7: Build User-Facing Output**

  - [ ] **Subtask 7.1**: Show intermediate thinking process by outputting messages for each major step the agent does to the discord user
  - [ ] **Subtask 7.2**: Ensure the output includes short rationales and reference links
  - [ ] **Subtask 7.3**: Add an option to refine or iterate on the research direction

- [ ] **Task 8: Testing & Validation**

  - [ ] **Subtask 8.1**: Conduct end-to-end tests with various research inputs
  - [ ] **Subtask 8.2**: Assess accuracy, clarity, and speed of the pipeline
  - [ ] **Subtask 8.3**: Collect feedback for any identified gaps or errors

- [ ] **Task 9: Documentation & Deployment**
  - [ ] **Subtask 9.1**: Document each Mistral function and Perplexity Sonar Pro usage
  - [ ] **Subtask 9.2**: Provide a quick-start guide for developers and users
  - [ ] **Subtask 9.3**: Deploy the system with environment-specific configurations
