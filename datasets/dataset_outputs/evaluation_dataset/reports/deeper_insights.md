# Deeper Insights Report

## Dataset Inventory
- **Unique HuggingFace datasets**: 7
- **Total source specs**: 11
  - `rajpurkar/squad_v2` — Reading comprehension questions from SQuAD v2
  - `yahma/alpaca-cleaned` — Cleaned instruction-following prompts from Alpaca
  - `JailbreakBench/JBB-Behaviors` — Curated harmful and benign behavior goals from JailbreakBench
  - `lmsys/toxic-chat` — Real-world toxic user inputs from LMSYS Chatbot Arena, filtered by jailbreaking and toxicity flags
  - `TrustAIRLab/in-the-wild-jailbreak-prompts` — Real-world jailbreak prompts and regular prompts collected in the wild
  - `rubend18/ChatGPT-Jailbreak-Prompts` — Community-sourced ChatGPT jailbreak prompts (role-play style)
  - `JailbreakV-28K/JailBreakV-28k` — Large-scale jailbreak benchmark with 28K attack prompts across 7 formats (Template, SD, figstep, typo, Persuade, Logic, SD_typo) and 16 policy categories; includes both jailbreak queries and underlying red-team queries

## Core Summary
- Total prompts: 20137
- Class distribution: {'jailbreak': 9662, 'harmful': 2356, 'benign': 8119}
- Split distribution: {'validation': 3017, 'train': 14093, 'test': 3027}
- Jailbreak attack distribution: {'instruction_override': 4875, 'multi_step': 170, 'obfuscation': 3643, 'role_play': 801, 'prompt_injection': 173}
- Jailbreak attack entropy (bits): 1.5328

## Source Contribution
- `jailbreakv28k_jailbreak`: 8641 (42.9%)
- `alpaca_instructions`: 3496 (17.4%)
- `squad_v2_questions`: 3485 (17.3%)
- `jailbreakv28k_redteam`: 1996 (9.9%)
- `trustairlab_regular_benign`: 1038 (5.2%)
- `trustairlab_in_the_wild_jailbreak`: 784 (3.9%)
- `toxic_chat_harmful`: 260 (1.3%)
- `toxic_chat_jailbreak_variation`: 108 (0.5%)
- `toxic_chat_jailbreak`: 108 (0.5%)
- `jbb_behaviors_benign`: 100 (0.5%)
- `jbb_behaviors_harmful`: 100 (0.5%)
- `trustairlab_in_the_wild_jailbreak_variation`: 12 (0.1%)
- `rubend18_chatgpt_jailbreak_prompts`: 9 (0.0%)

## JailBreakV-28K Analysis
- Total JBV records: 10637
- By label: {'jailbreak': 8641, 'harmful': 1996}
- By attack type: {'instruction_override': 4584, 'obfuscation': 3640, 'role_play': 342, 'prompt_injection': 74, 'multi_step': 1}
- By JBV format: {'Template': 5742, 'SD_typo': 1820, 'SD': 1820, 'figstep': 751, 'Persuade': 421, 'Logic': 83}
- By JBV policy (top): {'Malware': 1172, 'Illegal Activity': 1089, 'Fraud': 951, 'Bias': 823, 'Economic Harm': 807, 'Government Decision': 680, 'Unethical Behavior': 662, 'Hate Speech': 593, 'Violence': 584, 'Physical Harm': 563}
- Attack type entropy (bits): 1.2553

## Length Insights
- Global length stats: {'min': 12, 'max': 2500, 'mean': 361.65, 'median': 109, 'p90': 1188.0, 'p95': 1676.2}
- Length by class: {'jailbreak': {'count': 9662, 'mean': 608.33, 'median': 421.0, 'p90': 1631.0}, 'harmful': {'count': 2356, 'mean': 93.97, 'median': 70.0, 'p90': 132.0}, 'benign': {'count': 8119, 'mean': 145.75, 'median': 58, 'p90': 321.4}}

## Balance and Split Integrity
- Benign vs attack ratio: {'benign': 8119, 'attack_total': 12018, 'attack_to_benign': 1.4802}
- Global class ratios: {'jailbreak': 0.479813, 'harmful': 0.116999, 'benign': 0.403188}
- Split class ratios: {'train': {'jailbreak': 0.479742, 'harmful': 0.117008, 'benign': 0.40325}, 'validation': {'jailbreak': 0.479616, 'harmful': 0.117004, 'benign': 0.403381}, 'test': {'jailbreak': 0.480344, 'harmful': 0.116947, 'benign': 0.402709}}
- Split ratio drift: {'train': {'jailbreak': 7.2e-05, 'harmful': 1e-05, 'benign': 6.2e-05}, 'validation': {'jailbreak': 0.000198, 'harmful': 5e-06, 'benign': 0.000193}, 'test': {'jailbreak': 0.00053, 'harmful': 5.1e-05, 'benign': 0.000479}}

## Source Concentration
- Top sources by label: {'jailbreak': [('jailbreakv28k_jailbreak', 8641), ('trustairlab_in_the_wild_jailbreak', 784), ('toxic_chat_jailbreak_variation', 108), ('toxic_chat_jailbreak', 108), ('trustairlab_in_the_wild_jailbreak_variation', 12)], 'harmful': [('jailbreakv28k_redteam', 1996), ('toxic_chat_harmful', 260), ('jbb_behaviors_harmful', 100)], 'benign': [('alpaca_instructions', 3496), ('squad_v2_questions', 3485), ('trustairlab_regular_benign', 1038), ('jbb_behaviors_benign', 100)]}

## Actionable Notes
- Dataset draws from 7 unique Hugging Face datasets via 11 source specs.
- Class distribution is attack-heavy enough to stress false-negative detection while preserving benign coverage for false-positive checks.
- Split drift is low, indicating stable class proportions across train/validation/test.
- Jailbreak taxonomy coverage spans multiple strategies; monitor low-count attack types for threshold overfitting.
- Long-tail prompt lengths can impact model calibration; consider length-aware threshold diagnostics.
- JailBreakV-28K adds diverse attack formats (Template, SD, figstep, typo, Persuade, Logic, SD_typo) covering 16 safety-policy categories.
- JailBreakV-28K red-team queries provide the underlying harmful intents, complementing the jailbreak-wrapped variants for balanced harmful-class coverage.