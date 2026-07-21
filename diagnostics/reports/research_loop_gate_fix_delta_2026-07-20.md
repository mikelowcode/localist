# Research Loop Gate Fix — Before/After Delta, 2026-07-20

**Fix:** `_evaluate_pricing_gate()` (`backend/mcp_tool_dispatcher.py`) now takes `instruction` alongside `text` and judges whether the text specifically answers the question asked, not just whether pricing/spec-shaped content is present anywhere. See `diagnostics/reports/research_loop_qa_assessment_2026-07-20.md` for the false-positive pattern this targets.

**Runtime:** backend=`ollama` chat_model=`gemma4:31b-cloud` (LIVE, no mocks) — confirmed before running, same check as the original pass.

**Before data source:** `diagnostics/research_loop_qa_2026-07-20.csv` (original pass, read directly, not re-derived from the report's prose).

**After data source:** `diagnostics/research_loop_gate_fix_2026-07-20.csv` (this re-run).

## Scope note

Only the 6 flagged queries below were re-run. The other 12 queries from the
original corpus (including Categories A/B/C's originally-correct immediate
passes) were **not** re-verified against the fixed gate in this pass — a
full 18-query re-run would be needed to rule out the tightened gate
introducing a new false-negative on a case that previously passed correctly
for the right reason. Flagged as a follow-up in §Summary below, not done here.

## Summary — before requesting to call this fix done

**4 of 6 improved, 2 of 6 unchanged, 0 of 6 regressed.** No case in this
batch shows a previously-correct pass now wrongly exhausting — i.e. no
over-correction found in this specific 6-query sample (see the scope note
above for why that's not the same as "no over-correction exists anywhere").

| # | Verdict |
|---|---|
| 2  (GitHub Copilot Individual) | **Improved** — was accepting an irrelevant credit-rate snippet; now fetches and finds real per-month prices ($10/$39/$100). |
| 11 (iPhone 16 price) | **No change** — identical snippet, identical gate outcome. The fix's "wrong tier/product" framing doesn't squarely address this case's actual problem (no single stated retail price, only trade-in/refurb deltas). |
| 12 (Salesforce contract price) | **No change** — identical snippet, identical gate outcome. The clearest remaining gap: the gate still equates a published list-tier price ("Enterprise $175/user/mo") with the "exact enterprise **contract** price" the question specifically asked about — a published-vs-negotiated distinction the current prompt wording doesn't clearly draw. |
| 14 (obscure NFT floor price) | **Partial improvement** — before: zero real numbers, pure definitional content; after: a real, sourced floor price for a real named collection. Still can't resolve the question's inherent unanswerability (no collection was named), but stopped accepting content with literally no pricing data in it at all. |
| 15 (F-150 Lightning payload) | **Improved** — before: fetched a page with zero payload data and passed anyway; after: gate-passed on a snippet (Wikipedia) stating the exact correct figure, 2,235 lb. Caveat: also took a different iteration path this run (1 iteration, no fetch, vs. 2 + fetch before) — live search results are non-deterministic, so this improvement isn't attributable to the gate fix alone with full confidence. |
| 17 ("worth the price", iPhone opinion) | **Improved — the cleanest win.** Before: reformulation quietly substituted a factual pricing query and gate-passed. After: reformulation still tries factual substitutions, but the gate (which judges relevance against the ORIGINAL instruction, not the reformulated one) correctly refuses to accept any of them as answering a value judgment, and the loop exhausts honestly. |

## Per-Query Delta

### Query 2 — What does GitHub Copilot Individual cost per month?

| | Before | After |
|---|---|---|
| Outcome | `gate_passed_on_snippet` | `gate_passed_after_fetch` |
| Iterations | 1 | 2 |
| Fetch occurred | False | True |
| Reformulated | False | True |
| Elapsed | 1.03s | 8.61s |

**Before answer:**
```
• GitHub Copilot · Plans & pricing
  Every plan includes a monthly allowance: <strong>1 AI credit = $0.01 USD</strong>. You use credits when you chat with Copilot, work with agents, or use Copilot CLI, Spaces, and Spark. Code completions and next edit suggestions don&#x27;t use credits.
  [https://github.com/features/copilot/plans]

• GitHub Copilot licenses - GitHub Docs
  You will receive a prorated refund for any remaining portion of your personal plan&#x27;s current billing cycle. You will then be able to continue using Copilot according to the policies set by your company. There are several ways to use Copilot for free. Provides limited access to Copilot features
  [https://docs.github.com/en/billing/concepts/product-billing/github-copilot-licenses]

• About individual GitHub Copilot plans and benefits - GitHub Docs
  Designed to give you a limited taste of Copilot&#x27;s capabilities ... Verified students can access unlimited completions and additional models at no cost. Includes
```
**After answer:**
```
Title: About individual GitHub Copilot plans and benefits - GitHub Docs
Source: https://docs.github.com/en/copilot/concepts/billing/individual-plans
Words: 417

GitHub offers multiple Copilot plans for individual developers, as well as a dedicated student offering, each designed to meet different needs based on your coding habits, interest in AI models, and desired level of flexibility. You can choose from the following plans. For developers looking to get started with Copilot. Includes up to 2,000 code completions and an allowance of GitHub AI Credits Limited chat and agent usage with models available through auto model selection only Designed to give you a limited taste of Copilot's capabilities No subscription or payment required Intended for personal use only, not for users managed by an organization or enterprise Great for developers who want to explore Copilot's capabilities before upgrading to a paid plan Verified students can access unlimited completions and additional models a
```
(full fetched text, not shown above, includes: "PlanPrice per month... Copilot Pro $10 USD... Copilot Pro+ $39 USD... Copilot Max $100 USD")

**Read: Improved.** The before-snippet only surfaced a per-AI-credit rate
("$0.01 USD") — never a monthly plan price — and the old gate accepted it
anyway. The new gate correctly said "no" on that same kind of content,
forced a reformulation + fetch, and landed on GitHub's own docs page, which
states real monthly prices for the individual-developer tiers (Pro $10,
Pro+ $39, Max $100). This is now a genuinely correct, specific answer where
before there was none.

### Query 11 — What's the price of an iPhone 16?

| | Before | After |
|---|---|---|
| Outcome | `gate_passed_on_snippet` | `gate_passed_on_snippet` |
| Iterations | 1 | 1 |
| Fetch occurred | False | False |
| Reformulated | False | False |
| Elapsed | 1.13s | 1.1s |

**Before answer:**
```
• Buy iPhone 16 and iPhone 16 Plus - Apple
  Get <strong>$35 - $695</strong> off a new iPhone 16 or iPhone 16 Plus when you trade in an iPhone 8 or newer. 0% financing available. Buy now with free shipping.
  [https://www.apple.com/shop/buy-iphone/iphone-16]

• iPhone 16 Price: How Much Should You Pay? — GHOSTEK
  The Pro line, including the iPhone ... cameras. However, these upgrades are heavily reflected in the price, costing <strong>$100 more than the previous model</strong>....
  [https://ghostek.com/blogs/ghostek-insider/iphone-16-price-how-much-should-you-pay]

• iPhone 16 Price In 2026: Here's How Much You Should Pay
  Generally speaking, the cost of a base model iPhone 16 comes in at <strong>around $564 for a Grade A condition model and tops out at around $750 for an iPhone 16 Pro Max</strong> (again, in Grade A condition).
  [https://www.knowyourmobile.com/phones/refurbished-smartphones/iphone-16-price/]
```
**After answer:**
```
• Buy iPhone 16 and iPhone 16 Plus - Apple
  Get <strong>$35 - $695</strong> off a new iPhone 16 or iPhone 16 Plus when you trade in an iPhone 8 or newer. 0% financing available. Buy now with free shipping.
  [https://www.apple.com/shop/buy-iphone/iphone-16]

• iPhone 16 Price: How Much Should You Pay? — GHOSTEK
  The Pro line, including the iPhone ... cameras. However, these upgrades are heavily reflected in the price, costing <strong>$100 more than the previous model</strong>....
  [https://ghostek.com/blogs/ghostek-insider/iphone-16-price-how-much-should-you-pay]

• iPhone 16 Price In 2026: Here's How Much You Should Pay
  Generally speaking, the cost of a base model iPhone 16 comes in at <strong>around $564 for a Grade A condition model and tops out at around $750 for an iPhone 16 Pro Max</strong> (again, in Grade A condition).
  [https://www.knowyourmobile.com/phones/refurbished-smartphones/iphone-16-price/]
```

**Read: No change.** Byte-identical snippet, byte-identical gate decision
(1 iteration, no fetch, "yes" on the first try both times). None of the
three results states iPhone 16's actual current base retail price ($799,
per query 17's search results elsewhere in this corpus) — only a trade-in
discount range, a relative delta vs. the previous model, and refurbished-
market prices. The new prompt's "different tier/trim than the one asked
about" framing doesn't clearly cover this failure shape (wrong *price type*
— discount/refurb/delta instead of retail — rather than wrong tier of the
same product), which is likely why it didn't move here. This is Category
D's ambiguity problem in a different guise than queries 12/2 and isn't
resolved by this fix.

### Query 12 — What is the exact enterprise contract price for Salesforce?

| | Before | After |
|---|---|---|
| Outcome | `gate_passed_on_snippet` | `gate_passed_on_snippet` |
| Iterations | 1 | 1 |
| Fetch occurred | False | False |
| Reformulated | False | False |
| Elapsed | 5.69s | 1.39s |

**Before answer:**
```
• Salesforce Pricing 2026: Plans from $25/User/Month
  Salesforce pricing: Starter $25, Pro Suite $100, <strong>Enterprise $175</strong>, Unlimited $350/user/mo. Free tier available. Verified from 11 sources, March 2026.
  [https://costbench.com/software/crm/salesforce/]

• Salesforce Pricing 2026: Plans, Add-Ons & Hidden Costs
  Salesforce pricing starts at <strong>$25/user/month</strong>, but real costs rise with API access, support, AI, security, and add-ons. See every 2026 plan.
  [https://saascrmreview.com/salesforce-pricing/]

• Salesforce Contracts Pricing | Salesforce
  This page is provided for information ... is part of the product.agentfoce Revenue Management product. It&#x27;s available for <strong>$50 per user per month</strong>....
  [https://www.salesforce.com/sales/revenue-lifecycle-management/contracts-pricing/]
```
**After answer:**
```
• Salesforce Pricing 2026: Plans from $25/User/Month
  Salesforce pricing: Starter $25, Pro Suite $100, <strong>Enterprise $175</strong>, Unlimited $350/user/mo. Free tier available. Verified from 11 sources, March 2026.
  [https://costbench.com/software/crm/salesforce/]

• Salesforce Pricing 2026: Plans, Add-Ons & Hidden Costs
  Salesforce pricing starts at <strong>$25/user/month</strong>, but real costs rise with API access, support, AI, security, and add-ons. See every 2026 plan.
  [https://saascrmreview.com/salesforce-pricing/]

• Salesforce Contracts Pricing | Salesforce
  This page is provided for information ... is part of the product.agentfoce Revenue Management product. It&#x27;s available for <strong>$50 per user per month</strong>....
  [https://www.salesforce.com/sales/revenue-lifecycle-management/contracts-pricing/]
```

**Read: No change — the clearest remaining gap.** Byte-identical snippet,
byte-identical gate decision. The question specifically asks for the
"exact enterprise **contract** price" — deliberately unlisted, negotiated
pricing — and the accepted content contains only published list-tier
prices ("Enterprise $175/user/mo"). The new prompt's wording ("a different
product, a different tier/trim... does NOT count") talks about tier/trim
mismatches, but "Enterprise" as a tier name is literally present in the
text, so a literal reading of the prompt plausibly still says "yes, this
addresses the Enterprise tier asked about" — missing the deeper
published-list-price vs. negotiated-contract-price distinction the
question actually turns on. This is a real, still-open gap: the fix
targets tier/product mismatches, not this specific "type of pricing"
mismatch. Worth a follow-up prompt revision if this exact question shape
matters.

### Query 14 — What is the current spot price of a specific obscure NFT collection's floor price?

| | Before | After |
|---|---|---|
| Outcome | `gate_passed_on_snippet` | `gate_passed_on_snippet` |
| Iterations | 1 | 2 |
| Fetch occurred | False | True |
| Reformulated | False | True |
| Elapsed | 1.11s | 5.92s |

**Before answer:**
```
• NFT Price Floor - NFT Collections Ranking by market cap, prices & trading stats
  Our NFT platform tracks the price floor of 1798 NFT collections with a total trading volume of <strong>3,113,238 USD</strong> and 31593 total sales over the last day.
  [https://nftpricefloor.com/]

• Top NFT Collections by Market Cap: Floor Price & Volume | CoinGecko
  NFT floor price is <strong>the lowest price an individual is willing to sell an NFT for</strong>. Alternatively, the NFT floor price can also be the lowest amount of ETH a person is able to spend to own an NFT or become a member of an NFT project. NFT floor price is often determined and set by the
  [https://www.coingecko.com/en/nft]

• Top NFT Collection Prices, Charts & Tracker | Forbes Digital Assets
  Forbes Digital Assets&#x27;s NFT collection tracker shows a summary of daily top NFT prices and charts by trading volume. Uncover the latest prices to inform your NFT strategy.
  [https://www.forbes.com/digital-assets/nft-prices/]
```
**After answer:**
```
• Blur: NFT Marketplace for Pro Traders
  Floor Price0.04 · 1D Volume0.65 · 1606 owners · Floor Price0.01 · 1D Volume0.01 · 3454 owners · Floor Price0.20 · 1D Volume0.00 · 2425 owners · Floor Price0.08 · 1D Volume0.79 · 605 owners · Floor Price0.31 · 1D Volume2.17 · Faster Sweeping · Marketplace fees ·
  [https://blur.io/]

• How can I explore my Portfolio? | OpenSea Help Center
  Highest movers are NFTs and tokens with the biggest price changes in the last day. This provides a list of all the collections that you own items from, the number of items held, the cumulative value of those items by collection, the floor price, the top offer, and the last 7D trend. This provides a
  [https://support.opensea.io/en/articles/12417631-how-can-i-explore-my-portfolio]

• Ξ 0.0018 SUPER SAPIENSS NFT | Blur
  SUPER SAPIENSS NFT #3914 · <strong>- 0.0018</strong> · 0.0009 · - 666AA2 · 1 · 15m ago · buy floor · 0.0018 · Optimize sweep · 1D · 1W · 1M · 0.0016 · 0.0017 · 0.0018 · 0.0019 · 0.0020 · 10 PM 
```

**Read: Partial improvement.** The before-content contained zero concrete
floor-price data for any actual collection — only a generic definition of
"what is NFT floor price" and an aggregate cross-collection statistic. The
new gate correctly rejected that and drove a reformulation + fetch, landing
on real, sourced data: an actual named collection ("SUPER SAPIENSS") with a
real floor price (0.0018 ETH) from a live marketplace. That said, the
question is inherently under-specified by design (it names no specific
collection — that's the point of the probe), so no gate, however
relevance-aware, can fully resolve "which obscure collection" was meant.
The improvement here is real but narrower than queries 2/15/17: the gate
now insists on *some* real, specific, sourced number rather than pure
definitional filler, but it cannot adjudicate a question that has no single
correct referent to begin with.

### Query 15 — What is the max payload capacity of a Ford F-150 Lightning?

| | Before | After |
|---|---|---|
| Outcome | `gate_passed_after_fetch` | `gate_passed_on_snippet` |
| Iterations | 2 | 1 |
| Fetch occurred | True | False |
| Reformulated | True | False |
| Elapsed | 15.8s | 1.57s |

**Before answer:**
```
Title: 2025 Ford F-150 Lightning Specs, Review, Price, & Trims | Sarchione Ford
Source: https://sarchioneford.com/ford-f-150-lightning-model-review
Words: 180

*Manufacturer’s Suggested Retail Price (also referred to as “MSRP”, “Base MSRP”, “Base Price” or the “Starting At” price), excludes destination/delivery charge, taxes, title, license, and registration and/or electronic filing fees, dealer fees, and total of options. For authenticated AXZ Plan customers, the price displayed may represent Plan pricing. Not all AXZ Plan customers will qualify for the Plan pricing shown and not all offers or incentives are available to AXZ Plan customers. Although every reasonable effort has been made to ensure the accuracy of the information contained on this site, absolute accuracy cannot be guaranteed. This site, and all information and materials appearing on it, are presented to the user "as is" without warranty of any kind, either express or implied. All vehicles are subject to prior sale. Pric
```
**After answer:**
```
• Ford F-150 Lightning - Wikipedia
  The <strong>Ford</strong> <strong>F</strong>-<strong>150</strong> <strong>Lightning</strong> was also evaluated to reach 62 mph (100 km/h) in 4.5 seconds. Maximum available <strong>payload</strong> is 2,235 pounds (1,014 kg), which includes the 400-pound (180 kg) <strong>payload</strong>
  [https://en.wikipedia.org/wiki/Ford_F-150_Lightning]

• Ford F-150 Lightning Towing Capacity: How Much Can It Pull?
  The Ford F-150 Lightning truck has a maximum towing capacity of <strong>10,000 pounds</strong>,1 while the gas-powered Ford F-150 has a maximum towing capacity of 14,000 pounds.1 Both the gas-powered and electric models of the Ford F-150 trucks are capable towing vehicles that have many towing
  [https://veteransfordtampa.com/ford-f-150-lightning-towing-capacity]

• 2025 Ford F-150 Lightning Review, Pricing, and Specs
  The maximum towing capacity of ... Range battery models are limited to 7700 pounds. Payload capacity is as high as <strong>2000 po
```

**Read: Improved, with a caveat.** Before: the loop fetched a full page
(180 words of pure MSRP/fees legal boilerplate, zero payload figures) and
the old gate said "yes" anyway. After: the Wikipedia snippet on the very
first try states the correct, specific figure — "Maximum available payload
is 2,235 pounds (1,014 kg)" — directly answering the question, and the new
gate correctly passed it. Caveat: this run also took a different mechanical
path than before (1 iteration/no fetch vs. 2 iterations/fetch previously)
— live search results vary run to run, so part of this improvement may
reflect Brave returning better results this time rather than the gate fix
alone. What the gate fix can still be credited for: unlike before, if this
run *had* landed on the same payload-free legal-boilerplate page, the new
relevance-aware prompt gives real reason to expect a "no" rather than
another false "yes" — that specific claim isn't tested by this particular
run, though.

### Query 17 — Do you think the new iPhone is worth the price?

| | Before | After |
|---|---|---|
| Outcome | `gate_passed_on_snippet` | `exhausted_honestly` |
| Iterations | 2 | 3 |
| Fetch occurred | True | True |
| Reformulated | True | True |
| Elapsed | 3.79s | 9.91s |

**Before answer:**
```
• Buy iPhone 16 and iPhone 16 Plus - Apple
  Get <strong>$35 - $695</strong> off a new iPhone 16 or iPhone 16 Plus when you trade in an iPhone 8 or newer. 0% financing available. Buy now with free shipping.
  [https://www.apple.com/shop/buy-iphone/iphone-16]

• iPhone 16 Release Price Guide: What to Pay & When
  If you’re a typical user, you don’t need to overthink this: the <strong>iPhone 16 (128GB) at $699</strong> — now widely available at Apple’s official store and major carriers — delivers the strongest balance of feature access, longevity, and cost in ...
  [https://electronics.alibaba.com/buyingguides/iphone-16-price-guide-when-to-buy-which-model]

• iPhone 16 USA Price Guide: How Much to Pay & Where to Buy
  <strong>New, unlocked, Apple Store: $799 (16, 128GB) → $1,199 (Pro Max, 512GB) Refurbished (Back Market): $629–$699 (16, 128–256GB)</strong> 4 · Used, unlocked (Swappa): $443 (16, 128GB, Fair condition) → $680 (16, 256GB, Excellent) 2 · Carrier deal ...
  [https://electroni
```
**After answer:**
```
ERROR: research loop exhausted 3 iteration(s) without finding concrete pricing information (queries tried: ['Do you think the new iPhone is worth the price?', 'iPhone 16 pricing plans and official cost by model', 'iPhone 16 official price list by storage capacity apple.com']).
```

**Read: Improved — the cleanest and most important win in this batch.**
Before, reformulation quietly rewrote the subjective question into a
factual pricing query and the loop reported success on real iPhone prices
($699/$799) — technically true numbers, but they never actually answer "is
it worth the price" (a value judgment, not a lookup). After: reformulation
still tries the same kind of factual substitution (its own two rewrites,
visible in `queries_tried`, are both plain factual price queries), but the
gate now checks relevance against the ORIGINAL instruction text — which is
still the subjective question, unchanged throughout the loop — and
correctly refuses to accept any factual pricing content as answering it.
The loop exhausts honestly after all 3 iterations, which is exactly the
outcome Category G was designed to probe for. This also demonstrates a
useful design property confirmed by this run: the gate's relevance check
is anchored to the original instruction, not to whatever the reformulated
query narrowed to — which is what makes this case work correctly.
