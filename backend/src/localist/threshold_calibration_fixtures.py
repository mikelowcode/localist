"""
Shared test-battery fixtures for semantic-gating threshold calibration.

Single source of truth for the utterance batteries used to measure
planner.py's four semantic-gating thresholds (explicit_search_action,
lookup_request, research_intent, episodic_relevance). Read by both:
  - diagnostics/nomic_embed_text_threshold_probe.py (offline, human-reviewed)
  - threshold_calibration.py (live, in-app calibration)

Also the episodic recall floor battery (RECALL_*, bottom of this file), read
by diagnostics/diag_embeddinggemma_episodic_recall_floor.py and
threshold_calibration.calibrate_recall_floor().

These must never drift apart, since a discrepancy would make live-calibrated
thresholds not comparable to the hand-validated ones. Data copied verbatim
from the original tuning diagnostics (score_lookup_request_templates.py,
score_research_intent_templates.py, score_episodic_relevance_templates.py) —
do not re-invent or rephrase entries here; add new ones instead if coverage
needs to grow.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# lookup_request / explicit_search_action — Categories A-D
# ---------------------------------------------------------------------------

LR_CAT_A: list[tuple[str, str]] = [
    ("A", "Tell me how Localist works?"),
    ("A", "Can you read my wiki files?"),
    ("A", "List the files in my vault?"),
]

LR_CAT_B: list[tuple[str, str]] = [
    ("B-identity", "who are you"),
    ("B-identity", "what are you"),
    ("B-identity", "what can you do"),
    ("B-identity", "what can you help with"),
    ("B-identity", "what do you do"),
    ("B-greeting", "hey lora"),
    ("B-greeting", "hi there"),
    ("B-greeting", "hey there"),
    ("B-greeting", "what's up"),
]

LR_CAT_C: list[tuple[str, str]] = [
    ("C", "Can you look up Apple's price hike for the MacBook Neo and iPad?"),
    ("C", "Can you look up their next-generation in-house Microsoft AI models?"),
    ("C", "Can you look up Microsoft's next-generation in-house AI models?"),
]

LR_CAT_D: list[tuple[str, str, str]] = [
    ("D-verb-swap",   "short/generic",        "Can you help me with this?"),
    ("D-verb-swap",   "short/generic",        "Could you check this for me?"),
    ("D-verb-swap",   "short/generic",        "Would you look at this?"),
    ("D-verb-swap",   "short/generic",        "Can you tell me about this?"),
    ("D-modal-swap",  "short/generic",        "Will you look into this?"),
    ("D-modal-swap",  "short/generic",        "Do you mind looking at this?"),
    ("D-length",      "bare-minimum",         "Can you help?"),
    ("D-length",      "fuller-sentence",      "Can you help me understand this particular concept in more depth?"),
    ("D-domain",      "project-referential",  "Can you help me understand how Localist works?"),
    ("D-domain",      "project-referential",  "Could you explain what this system does?"),
    ("D-domain",      "file-referencing",     "Can you look at my notes and help me organize them?"),
    ("D-domain",      "file-referencing",     "Could you read through this document for me?"),
    ("D-domain",      "generic/unrelated",    "Would you help me plan a trip to Japan?"),
    ("D-domain",      "generic/unrelated",    "Can you tell me a joke?"),
]

# ---------------------------------------------------------------------------
# research_intent — Categories T/L/K/F/E/G
# ---------------------------------------------------------------------------

RI_CAT_T: list[str] = [
    "How much does the Tesla Model 3 cost?",
    "What's the price of the new iPhone?",
    "Can you find out how much AWS charges for S3 storage?",
    "What are the pricing tiers for Notion?",
    "Look up the specs on the RTX 4090",
    "Find me the cost of a one-bedroom apartment in Austin",
    "What does ChatGPT Plus cost per month?",
    "Track down pricing for Salesforce Enterprise",
    "Can you check what a Peloton subscription costs?",
    "What's the going rate for a plumber in this area?",
]
RI_CAT_L: list[str] = [
    "Can you look up the release date for this?",
    "Could you look up what year this happened?",
    "Can you look up information about the latest Apple products?",
    "Can you look up Apple's price hike for the MacBook Neo and iPad?",
    "Can you look up their next-generation in-house Microsoft AI models?",
]
RI_CAT_L_PRICE_ADJACENT: str = "Could you find out the current stock price for me?"
RI_CAT_K: list[str] = [
    "What is this?",
    "Tell me about this company",
    "Explain how blockchain works",
    "What do you know about this?",
]
RI_CAT_F: list[str] = [
    "What's the latest on this?",
    "Is there anything new about this?",
    "What's the current status of this project?",
]
RI_CAT_E: list[str] = [
    "Is this too expensive for me?",
    "Do you think this is worth the price?",
    "I can't believe how expensive rent is these days",
    "That seems like a lot of money for what you get",
]
RI_CAT_G: list[str] = [
    "Can you help me with this?",
    "What's up?",
    "How are you doing today?",
    "Thanks, that's helpful",
]

# ---------------------------------------------------------------------------
# episodic_relevance — POSITIVES/NEGATIVES
# ---------------------------------------------------------------------------

EP_POSITIVES: list[str] = [
    "Help me prepare for the upcoming Claude Impact Lab on August 6th.",
    "What do I need to bring to my dentist appointment next week?",
    "Can you help me get ready for my presentation on Friday?",
    "What did we decide about the migration plan?",
    "What do I have going on this week?",
    "Catch me up on my project status.",
    "What was the plan we settled on last time?",
    "Help me get ready for my trip to Japan.",
    "Prep me for my meeting with the investors tomorrow.",
    "What did I say I would do about the server migration?",
]
EP_NEGATIVES: list[str] = [
    "What is the capital of France?",
    "Write a haiku about the ocean.",
    "Search the web for the latest news on SpaceX.",
    "What is 2+2?",
    "Summarize this document for me.",
    "Fetch this URL and tell me what it says.",
    "Remind me how photosynthesis works.",
    "Help me write a cover letter for a job application.",
    "Prepare a report on quarterly earnings.",
    "What is on the front page of the New York Times today?",
    "Explain how neural networks work.",
    "Get ready to receive a large file upload.",
    "I have a headache, what should I do?",
    "My favorite color is blue, what goes well with it?",
    "Can you help me plan a birthday party for my friend?",
    "What is my IP address?",
    "Prepare a Python script that reverses a string.",
    "I need help understanding this error message.",
    "What time is it in Tokyo right now?",
    "Draft an email to my landlord about the leak.",
]

# A real live-reported failure (Xbox/gaming episodic-memory miss against
# nomic-embed-text), added on top of the original battery.
EP_POSITIVES_LIVE: list[str] = [
    "What games do I play?",
    "What console do I own?",
    "Tell me about my Xbox.",
]


# ---------------------------------------------------------------------------
# Episodic recall floor — the min_score for EpisodicMemoryReader.by_similarity()
# (docs/architecture/16-runtime-backend-layer.md §16.28). SYNTHETIC: made-up
# facts shaped like real episodes (one sentence starting "The user …";
# subject = content[:80]), with near-miss distractors in the same domains.
# Real episodes are personal data and never belong here. Positive queries
# are paraphrased full user instructions with one right episode; negatives
# have no relevant episode. Moved verbatim from the diagnostic that first
# measured it, so the hand-validated floors in planner._EPISODIC_RECALL_FLOORS
# and live calibration score the same battery.
# ---------------------------------------------------------------------------

RECALL_EPISODES: dict[str, tuple[str, str]] = {
    # key: (episode_type, content) — subject is derived as content[:80]
    "console":      ("preference",   "The user plays most games on a PlayStation 5 with a DualSense controller."),
    "fps_style":    ("preference",   "The user prefers aggressive close-range loadouts in shooters and plays with high stick sensitivity."),
    "racing":       ("preference",   "The user dislikes racing games and finds them repetitive."),
    "laptop":       ("project_fact", "The user develops on a 14-inch MacBook Pro with an M3 Pro chip and 36GB of unified memory."),
    "home_server":  ("project_fact", "The user runs a small Linux home server for backups and media streaming."),
    "search_api":   ("project_fact", "The user switched the assistant's web search provider from SerpAPI to Tavily."),
    "news_api":     ("project_fact", "The user's daily news brief pulls headlines from the NewsAPI developer tier."),
    "runtime":      ("project_fact", "The user runs local models through LM Studio and occasionally uses hosted models for long documents."),
    "context_cap":  ("project_fact", "The user capped the local model's context window at 16k tokens to keep memory usage stable."),
    "editor":       ("preference",   "The user writes code in Neovim with a minimal plugin setup."),
    "language":     ("preference",   "The user prefers Python for backend work and TypeScript for frontends."),
    "tabs":         ("naming_convention", "The user names feature branches with the prefix feat/ followed by a short kebab-case description."),
    "test_style":   ("naming_convention", "The user names test files test_<module>.py and keeps them in a top-level tests folder."),
    "coffee":       ("preference",   "The user drinks oat milk flat whites and avoids dairy."),
    "diet":         ("preference",   "The user is vegetarian but eats eggs."),
    "allergy":      ("project_fact", "The user is allergic to tree nuts."),
    "running":      ("preference",   "The user runs three mornings a week and is training for a half marathon in the spring."),
    "city":         ("project_fact", "The user lives in Portland, Oregon and commutes by bike."),
    "weather_unit": ("preference",   "The user wants temperatures shown in Celsius, not Fahrenheit."),
    "timezone":     ("project_fact", "The user works on Pacific time and usually starts at 7am."),
    "phone":        ("project_fact", "The user's phone is a Pixel 8 running the latest Android beta."),
    "brand_view":   ("preference",   "The user thinks Google's hardware naming has become confusing and should be simplified."),
    "music":        ("preference",   "The user listens to ambient electronic music while coding."),
    "reading":      ("preference",   "The user prefers short summaries with bullet points over long prose answers."),
    "tone":         ("preference",   "The user wants direct answers without filler or excessive caveats."),
    "api_dir":      ("project_fact", "The user recommends the GitHub public-apis list as a starting point for finding free APIs."),
    "deploy":       ("workflow",     "The user deploys the frontend to Netlify from the main branch and previews pull requests on separate URLs."),
    "db":           ("project_fact", "The user stores app data in SQLite with WAL mode enabled."),
    "pet":          ("project_fact", "The user has a border collie named Juniper."),
    "travel":       ("project_fact", "The user is planning a trip to Japan in October and wants to visit Kyoto."),
}

# (query, key of the one right episode). Paraphrased, indirect, and phrased
# as a full user instruction — never copying the episode's wording.
RECALL_POSITIVES: list[tuple[str, str]] = [
    ("What console do I usually play on?", "console"),
    ("Recommend a controller accessory that would work with my gaming setup.", "console"),
    ("Suggest a loadout for me in the new Call of Duty season.", "fps_style"),
    ("Should I buy the new Gran Turismo?", "racing"),
    ("How much RAM does my dev machine have?", "laptop"),
    ("Can my laptop run a 30B parameter model locally?", "laptop"),
    ("Where do my backups go?", "home_server"),
    ("Which web search service does my assistant use now?", "search_api"),
    ("Where do the headlines in my morning brief come from?", "news_api"),
    ("What do I use to run local LLMs?", "runtime"),
    ("Why does my local model forget things after long conversations?", "context_cap"),
    ("Set up a good config for my text editor.", "editor"),
    ("Which language should I write this new API service in?", "language"),
    ("What should I call the branch for the login page work?", "tabs"),
    ("Where should the tests for the parser module go?", "test_style"),
    ("Order me my usual coffee.", "coffee"),
    ("Find me a dinner recipe for tonight.", "diet"),
    ("Is this granola safe for me to eat? It has almonds.", "allergy"),
    ("Build me a training plan for the next few weeks.", "running"),
    ("Find a good bike route for my commute.", "city"),
    ("What's the weather going to be like this weekend?", "weather_unit"),
    ("Schedule a call for early tomorrow my time.", "timezone"),
    ("Why does my phone keep crashing after the update?", "phone"),
    ("What do you think of the Pixel lineup's product names?", "brand_view"),
    ("Make me a playlist for a focused coding session.", "music"),
    ("Explain how transformers work.", "reading"),
    ("Where can I find a free weather API for my side project?", "api_dir"),
    ("How do I ship the latest frontend changes?", "deploy"),
    ("What database does my app use?", "db"),
    ("What's a good name for a toy for my dog?", "pet"),
    ("What should I pack for my upcoming trip?", "travel"),
]

# Questions with no relevant episode — anything scoring >= floor here would
# be injected into the prompt as irrelevant memory.
RECALL_NEGATIVES: list[str] = [
    "What is the capital of Australia?",
    "Summarize the causes of the French Revolution.",
    "Convert 250 grams to ounces.",
    "Write a haiku about autumn leaves.",
    "How do vaccines train the immune system?",
    "Explain the difference between TCP and UDP.",
    "Who won the 1998 World Cup?",
    "What's the derivative of x squared times sine x?",
    "Translate 'good morning' into Spanish.",
    "How many moons does Jupiter have?",
    "Give me a regex that matches an email address.",
    "What year did the Berlin Wall fall?",
    "Explain compound interest with an example.",
    "What is photosynthesis?",
    "List three famous paintings by Monet.",
]
