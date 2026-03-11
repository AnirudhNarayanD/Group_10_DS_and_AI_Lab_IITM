# Deeper Insights Report

## Core Summary
- Total prompts: 1500
- Class distribution: {'benign': 670, 'jailbreak': 455, 'harmful': 375}
- Split distribution: {'train': 1045, 'test': 233, 'validation': 222}
- Jailbreak attack distribution: {'role_play': 205, 'instruction_override': 124, 'multi_step': 84, 'prompt_injection': 41, 'obfuscation': 1}
- Jailbreak attack entropy (bits): 1.8116

## Length Insights
- Global length stats: {'min': 15, 'max': 2490, 'mean': 464.18, 'median': 92.0, 'p90': 1534.1, 'p95': 1874.2}
- Length by class: {'benign': {'count': 670, 'mean': 165.88, 'median': 58.0, 'p90': 422.0}, 'jailbreak': {'count': 455, 'mean': 1127.17, 'median': 1109, 'p90': 2075.8}, 'harmful': {'count': 375, 'mean': 192.71, 'median': 89, 'p90': 544.6}}

## Balance and Split Integrity
- Benign vs attack ratio: {'benign': 670, 'attack_total': 830, 'attack_to_benign': 1.2388}
- Global class ratios: {'benign': 0.446667, 'jailbreak': 0.303333, 'harmful': 0.25}
- Split class ratios: {'train': {'benign': 0.447847, 'jailbreak': 0.301435, 'harmful': 0.250718}, 'validation': {'benign': 0.45045, 'jailbreak': 0.297297, 'harmful': 0.252252}, 'test': {'benign': 0.437768, 'jailbreak': 0.317597, 'harmful': 0.244635}}
- Split ratio drift: {'train': {'benign': 0.00118, 'jailbreak': 0.001898, 'harmful': 0.000718}, 'validation': {'benign': 0.003784, 'jailbreak': 0.006036, 'harmful': 0.002252}, 'test': {'benign': 0.008898, 'jailbreak': 0.014263, 'harmful': 0.005365}}

## Source Concentration
- Top sources by label: {'benign': [('squad_v2_questions', 281), ('alpaca_instructions', 280), ('trustairlab_regular_benign', 102), ('jbb_behaviors_benign', 7)], 'jailbreak': [('trustairlab_in_the_wild_jailbreak', 328), ('toxic_chat_jailbreak_variation', 49), ('toxic_chat_jailbreak', 47), ('trustairlab_in_the_wild_jailbreak_variation', 27), ('rubend18_chatgpt_jailbreak_prompts', 4)], 'harmful': [('toxic_chat_harmful', 260), ('jbb_behaviors_harmful', 100), ('fallback_harmful', 15)]}

## Actionable Notes
- Class distribution is attack-heavy enough to stress false-negative detection while preserving benign coverage for false-positive checks.
- Split drift is low, indicating stable class proportions across train/validation/test.
- Jailbreak taxonomy coverage exists across multiple strategies; monitor low-count attack types for threshold overfitting.
- Long-tail prompt lengths can impact model calibration; consider length-aware threshold diagnostics.