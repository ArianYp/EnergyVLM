# Automated pre-screen of the counterfactual edits

5042 records. Baseline reversal rate 0.2723.

| flag | count | share | reversal rate | clean rate | difference | 95% CI | associated? |
|---|---:|---:|---:|---:|---:|---|:-:|
| `no_op_edit` | 0 | 0.00% | — | 0.2723 | +nan | [+nan, +nan] | no |
| `target_absent` | 0 | 0.00% | — | 0.2723 | +nan | [+nan, +nan] | no |
| `article_introduced` | 185 | 3.67% | 0.1784 | 0.2759 | -0.0975 | [-0.1533, -0.0400] | yes |
| `multi_token_change` | 152 | 3.01% | 0.5461 | 0.2638 | +0.2822 | [+0.2019, +0.3594] | yes |

## How to read the association column

Association with the reversal flag is not the same as confounding. Of the two flags that fire:

- **`article_introduced`** (185 records, 3.7%, all colour/shape) is a genuine surface defect: the substitution rule produces *"a orange chair"*. It is **anti**-correlated with reversal, so it dilutes rather than drives the sharp subset. Report as a limitation; it does not threaten identification.
- **`multi_token_change`** (152 records, 3.0%) is **not** a defect. 151 of 152 are `spatial` edits, where a single relation atom legitimately spans several tokens (*"far from"* → *"on side of"*). Its association with reversal is the already-known fact that `spatial` has the highest reversal rate of any category (57.7%), re-expressed at token level. The human audit adjudicates atom count; token count cannot.

> multi_token_change is not necessarily a defect: a single spatial-relation atom can span several tokens ('far from' -> 'on side of'). It is reported so the human audit can adjudicate, which is why the review sheet asks about atoms rather than tokens.

## Examples

**`article_introduced`**

- `a brown bird and a red giraffe` → `a orange bird and a red giraffe`
- `a brown banana and a green sheep` → `a orange banana and a green sheep`
- `a brown horse and a white cow` → `a orange horse and a white cow`
- `a brown banana and a green bird` → `a orange banana and a green bird`
- `a brown chair and a red clock` → `a orange chair and a red clock`

**`multi_token_change`**

- `a fish next to a bicycle` → `a fish far from a bicycle`
- `a sheep next to a wallet` → `a sheep far from a wallet`
- `a woman next to a bird` → `a woman far from a bird`
- `a frog next to a clock` → `a frog far from a clock`
- `a boy next to a mouse` → `a boy far from a mouse`

