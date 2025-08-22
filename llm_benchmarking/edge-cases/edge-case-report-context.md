# Contextual Measures Edge Case Analysis Report

*Generated on: 2025-08-22 08:48:56*

*This report focuses on Faithfulness, Contextual Precision, and Contextual Recall measures for LLM data only.*

## Overview

- **Total Contextual Edge Cases Analyzed**: 120
- **Data Source**: LLM data only
- **Measures Analyzed**: Faithfulness, Contextual Precision, Contextual Recall
- **Question Types**: bee_species, pesticides, additional_stressors, experimental_methodology, significance, future_research, limitations

## Overall Statistics

- **Total Cases**: 120
- **Score Range**: 0.000 - 0.722
- **Average Score**: 0.425

## Question Type Analysis

### Bee Species

- **Number of Cases**: 20
- **Score Range**: 0.333 - 0.361
- **Average Score**: 0.335

**LLM Analysis:**

**Section 1: Main Issues**

1. **Lack of Specificity in Contextual Precision**: Many nodes fail to specify the exact bee species tested, leading to irrelevant retrievals. This results in low precision scores as the nodes do not directly address the input question about specific bee species.
   
2. **Inadequate Contextual Recall**: The expected outputs often cannot be linked to any nodes in the retrieval context, indicating a complete absence of relevant information about the specific bee species mentioned in the queries.

3. **Misranking of Relevant Information**: Relevant nodes, when present, are often ranked lower than irrelevant ones, further contributing to low precision scores and ineffective retrieval of pertinent information.

**Section 2: Recommendations for Improving Prompt**

1. **Enhance Specificity in Queries**: Clearly specify the required bee species in the prompt to ensure retrieval contexts focus on relevant nodes that mention these species explicitly.

2. **Improve Node Ranking Algorithms**: Adjust the ranking system to prioritize nodes that directly mention the specific bee species in question, ensuring relevant information is ranked higher.

3. **Expand Retrieval Contexts**: Broaden the retrieval context to include more comprehensive data sources that explicitly mention various bee species, improving both precision and recall.

**Sample Contextual Edge Cases:**

1. **paper_586_case_19** (Paper 586)
   - Contextual Combined Score: 0.333
   - Input: What species of bee(s) were tested?
   - **Expected Output:** The bee species tested was Bombus terrestris. The subspecies was Bombus terrestrus audax.
   - **Actual Output:** Honeybees and bumblebees were tested.

2. **paper_584_case_37** (Paper 584)
   - Contextual Combined Score: 0.333
   - Input: What species of bee(s) were tested?
   - **Expected Output:** *Bombus terrestris audax*
   - **Actual Output:** *Bombus terrestris audax*

3. **paper_299_case_60** (Paper 299)
   - Contextual Combined Score: 0.333
   - Input: What species of bee(s) were tested?
   - **Expected Output:** Apis mellifera carnica
   - **Actual Output:** The species of bees tested were Halictus tumulorum, Lasioglossum fulvicorne, L. malachurum, L. pauxillum, Osmia spinulosa, honey bees (*Apis mellifera...

---

### Pesticides

- **Number of Cases**: 20
- **Score Range**: 0.222 - 0.706
- **Average Score**: 0.505

**LLM Analysis:**

**Section 1: Main Issues**
1. **Contextual Precision**: Many responses scored low due to irrelevant nodes being ranked higher than relevant ones. Nodes often lacked specific details about pesticides, such as their doses, exposure methods, or durations, leading to a complete absence of pertinent information.
2. **Contextual Recall**: Low scores were frequently due to the expected output containing details not supported by the retrieval context. This included missing specifics on pesticide application, exposure methods, and testing conditions.
3. **Faithfulness**: Some responses inaccurately represented the retrieval context, either by including incorrect details about pesticide usage or misclassifying substances, leading to discrepancies between the output and the context.

**Section 2: Recommendations for Improving Prompt**
1. **Enhance Contextual Relevance**: Ensure the prompt emphasizes the need for specific details about pesticides, such as their types, doses, and exposure methods, to improve the relevance of retrieved nodes.
2. **Prioritize Relevant Nodes**: Adjust the prompt to prioritize nodes that directly address the input query, ensuring that relevant information is ranked higher than irrelevant content.
3. **Clarify Expected Output**: Clearly define the expected output in the prompt to align with the retrieval context, reducing discrepancies and improving faithfulness in responses.

**Sample Contextual Edge Cases:**

1. **paper_382_case_109** (Paper 382)
   - Contextual Combined Score: 0.222
   - Input: What pesticide(s) were used in this study, and what was the dose, exposure method and duration of ex...
   - **Expected Output:** No pesticides were directly applied in the study. Instead, the study involved placing hives at sites within commercial orchards. However, pesticide ap...
   - **Actual Output:** The pesticides used in the study were neonicotinoids, specifically imidacloprid, clothianidin, and thiamethoxam. The exposure method for imidacloprid ...

2. **paper_380_case_125** (Paper 380)
   - Contextual Combined Score: 0.310
   - Input: What pesticide(s) were used in this study, and what was the dose, exposure method and duration of ex...
   - **Expected Output:** No exposure was conducted in the study. The study identified the pesticide residues present in the honey bee body.
   - **Actual Output:** The pesticides used in the study include Propamocarb (fungicide), Carbendazim (herbicide), and Azobenzene (insecticide). However, the study does not s...

3. **paper_585_case_31** (Paper 585)
   - Contextual Combined Score: 0.333
   - Input: What pesticide(s) were used in this study, and what was the dose, exposure method and duration of ex...
   - **Expected Output:** The pesticides used in the study were imidacloprid (50 ppb) and thiacloprid (4500 ppb). The exposure method involved mixing the pesticides in sugar sy...
   - **Actual Output:** The pesticides used in the study were imidacloprid (50 ppb) and thiacloprid (4500 ppb). The exposure method involved mixing the pesticides in sugar sy...

---

### Additional Stressors

- **Number of Cases**: 20
- **Score Range**: 0.000 - 0.333
- **Average Score**: 0.223

**LLM Analysis:**

**Section 1: Main Issues**

- **Irrelevant Contexts**: The most prevalent issue is the retrieval of irrelevant contexts, with nodes focusing on topics unrelated to additional stressors, leading to a Contextual Precision score of 0.00 in many cases.
- **Misalignment with Retrieval Context**: Many responses inaccurately represent or omit details from the retrieval context, affecting Faithfulness scores. Common errors include incorrect dosage levels, exposure durations, and missing stressors.
- **Lack of Contextual Recall**: Responses frequently fail to align with any nodes in the retrieval context, resulting in a Contextual Recall score of 0.00. This indicates a complete mismatch between the expected output and the retrieved information.

**Section 2: Recommendations for Improving Prompt**

- **Enhance Context Relevance**: Refine retrieval algorithms to prioritize nodes that directly address additional stressors, ensuring that relevant information is ranked higher than irrelevant content.
- **Clarify Output Requirements**: Specify the need for precise alignment with retrieval context details, such as dosage levels and exposure durations, to improve Faithfulness.
- **Improve Recall Alignment**: Design prompts to explicitly request confirmation of the presence or absence of additional stressors, aiding in better alignment with the retrieval context and improving Contextual Recall.

**Sample Contextual Edge Cases:**

1. **paper_515_case_182** (Paper 515)
   - Contextual Combined Score: 0.000
   - Input: Question about additional_stressors
   - **Expected Output:** No additional stressors were mentioned.
   - **Actual Output:** No additional stressors or combinations were mentioned in the text. The focus was solely on the effects of thiamethoxam.

2. **paper_787_case_45** (Paper 787)
   - Contextual Combined Score: 0.111
   - Input: Question about additional_stressors
   - **Expected Output:** No additional stressors were listed in the text
   - **Actual Output:** No additional stressors or combinations (like temperature, parasites, pathogens, other chemicals, or nutrition stress) were mentioned in the text. The...

3. **paper_443_case_91** (Paper 443)
   - Contextual Combined Score: 0.111
   - Input: Question about additional_stressors
   - **Expected Output:** No additional stressors were evaluated in the study.
   - **Actual Output:** No additional stressors or combinations (like temperature, parasites, pathogens, or nutritional stress) were mentioned in the texts. The only addition...

---

### Experimental Methodology

- **Number of Cases**: 20
- **Score Range**: 0.333 - 0.704
- **Average Score**: 0.579

**LLM Analysis:**

**Section 1: Main Issues**

1. **Contextual Precision**: Many responses suffer from low precision due to irrelevant nodes being ranked higher than relevant ones. This misalignment often results in the retrieval of nodes that do not address the specific experimental methodologies, leading to a lack of relevant content in the top-ranked nodes.

2. **Contextual Recall**: There is a frequent lack of alignment between the expected output and the retrieval context. Key details about experimental methodologies are often missing from the retrieval context, resulting in a failure to support the expected output with specific nodes.

3. **Faithfulness**: While generally high, some responses inaccurately include details not present in the retrieval context, such as incorrect species identification or unmentioned methodologies, slightly affecting the faithfulness scores.

**Section 2: Recommendations for Improving Prompt**

1. **Enhance Node Ranking**: Adjust the retrieval algorithm to prioritize nodes that directly address experimental methodologies, ensuring that relevant nodes are ranked higher than irrelevant ones.

2. **Improve Contextual Coverage**: Ensure that the retrieval context comprehensively covers all aspects of the expected output, including specific methodologies and experimental details, to improve recall.

3. **Clarify Methodology Focus**: Refine the prompt to explicitly request details on experimental methodologies, reducing the inclusion of irrelevant information and enhancing the precision of the responses.

**Sample Contextual Edge Cases:**

1. **paper_296_case_51** (Paper 296)
   - Contextual Combined Score: 0.333
   - Input: What experimental methodologies was used in this paper? (e.g. experiment done on the bees, not stati...
   - **Expected Output:** The experimental methodology used in this paper involved assessing the performance of bumble bees (*B. terrestris*) in queenless microcolonies, focusi...
   - **Actual Output:** The experimental methodology used in this paper involved assessing the performance of bumble bees (*B. terrestris*) in queenless microcolonies, focusi...

2. **paper_237_case_77** (Paper 237)
   - Contextual Combined Score: 0.333
   - Input: What experimental methodologies was used in this paper? (e.g. experiment done on the bees, not stati...
   - **Expected Output:** The authors employed free-flight assays. Sucrose feeders were provided in proximity to bee hives, and incrementally moved to a distance of 30m (the ex...
   - **Actual Output:** The experimental methodologies used in this paper include trials on free-ranging foragers making repeated flights over treated plots, flights to conta...

3. **paper_013_case_154** (Paper 013)
   - Contextual Combined Score: 0.333
   - Input: What experimental methodologies was used in this paper? (e.g. experiment done on the bees, not stati...
   - **Expected Output:** The experimental methodologies used in this paper include analyzing survival, pollen and syrup consumption, and midgut bacterial diversity in honey be...
   - **Actual Output:** The experimental methodologies used in this paper include analyzing survival, pollen and syrup consumption, and midgut bacterial diversity in honey be...

---

### Significance

- **Number of Cases**: 20
- **Score Range**: 0.333 - 0.722
- **Average Score**: 0.578

**LLM Analysis:**

**Section 1: Main Issues**
1. **Contextual Precision**: Many responses fail to prioritize relevant information, with irrelevant nodes often ranked higher than pertinent ones. This misalignment leads to low precision scores as the responses do not directly address the input's focus.
2. **Contextual Recall**: There is a frequent lack of alignment between the expected output and the retrieval context, with key details missing from the nodes. This results in low recall scores as the retrieval context does not support the expected findings.
3. **Faithfulness**: While generally high, occasional inaccuracies occur, such as misrepresenting study findings, which can slightly affect the overall evaluation.

**Section 2: Recommendations for Improving Prompt**
1. **Enhance Relevance Filtering**: Adjust the prompt to emphasize the importance of ranking nodes based on direct relevance to the input's focus, ensuring that pertinent information is prioritized.
2. **Improve Recall by Contextual Expansion**: Encourage the inclusion of broader retrieval contexts that cover all key aspects of the expected output, ensuring comprehensive support for the findings.
3. **Clarify Output Expectations**: Clearly define the expected output's focus and key details in the prompt to minimize misinterpretations and enhance alignment with the retrieval context.

**Sample Contextual Edge Cases:**

1. **paper_216_case_64** (Paper 216)
   - Contextual Combined Score: 0.333
   - Input: Summarize the paper's discussion regarding the importance to the field.
   - **Expected Output:** The authors present their significant finding to be the identification of Amelα1/ratβ2 gene targets for the synthesis of novel pesticides with low tox...
   - **Actual Output:** The paper underscores the importance of sulfoximines in agriculture for pest control, particularly their effectiveness against resistant insects. It a...

2. **paper_444_case_115** (Paper 444)
   - Contextual Combined Score: 0.333
   - Input: Summarize the paper's discussion regarding the importance to the field.
   - **Expected Output:** The main finding of the paper is that the performance of managed and wil pollinators are better in fields managed with the IPM strategy compared with ...
   - **Actual Output:** The paper emphasizes the importance of understanding the impact of neonicotinoid insecticides on both managed and wild pollinators, advocating for mor...

3. **paper_052_case_138** (Paper 052)
   - Contextual Combined Score: 0.333
   - Input: Summarize the paper's discussion regarding the importance to the field.
   - **Expected Output:** The study demonstrates that Apis cerana exhibits lower sensitivity to thiacloprid compared to imidacloprid. Thiacloprid exposure did not impair sucros...
   - **Actual Output:** The paper underscores the importance of studying the effects of neonicotinoids on *Apis cerana*, as this species is at risk from pesticide exposure, y...

---

### Future Research

- **Number of Cases**: 20
- **Score Range**: 0.286 - 0.333
- **Average Score**: 0.331

**LLM Analysis:**

**Section 1: Main Issues**

1. **Contextual Precision**: The primary issue is the ranking of irrelevant nodes higher than relevant ones, leading to a failure in addressing the input's focus on future research directions. This misalignment consistently results in low precision scores.

2. **Contextual Recall**: There is a complete mismatch between the expected output and the retrieval context, as the contexts do not contain any information on future research directions, leading to zero recall scores.

3. **Faithfulness**: Although not a primary issue in these cases, it is noteworthy that faithfulness scores are consistently high, indicating that when relevant information is present, the alignment is accurate and consistent.

**Section 2: Recommendations for Improving Prompt**

1. **Enhance Contextual Relevance**: Adjust the retrieval mechanism to prioritize nodes that explicitly mention future research directions, ensuring that relevant information is ranked higher.

2. **Improve Recall Mechanisms**: Implement strategies to better identify and extract future research suggestions from the context, possibly by refining keyword matching or using semantic analysis to detect implicit suggestions.

3. **Refine Node Ranking**: Develop a more sophisticated ranking algorithm that can discern the relevance of nodes based on the specific focus of the input, such as future research directions, to improve both precision and recall.

**Sample Contextual Edge Cases:**

1. **paper_440_case_78** (Paper 440)
   - Contextual Combined Score: 0.286
   - Input: Summarize the paper's discussion regarding future research directions.
   - **Expected Output:** The authors did not suggest directions for future research.
   - **Actual Output:** Future research should focus on the effects of imidacloprid on honeybee larvae, particularly examining the nursing behavior of bees and the potential ...

2. **paper_594_case_1** (Paper 594)
   - Contextual Combined Score: 0.333
   - Input: Summarize the paper's discussion regarding future research directions.
   - **Expected Output:** I cannot find any suggestions for future research.
   - **Actual Output:** Future research should investigate the intricate relationships between environmental factors and honey bee gene regulation, particularly under hive co...

3. **paper_586_case_19** (Paper 586)
   - Contextual Combined Score: 0.333
   - Input: Summarize the paper's discussion regarding future research directions.
   - **Expected Output:** I have not identified any sections where they discuss future research directions.
   - **Actual Output:** Future research should focus on the differential effects of neonicotinoids on receptor subtype activation, the long-term impacts on bee behavior and c...

---

---

*This report was generated automatically by the Contextual Measures Edge Case Analysis tool.*
