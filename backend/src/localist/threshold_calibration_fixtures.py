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


# ---------------------------------------------------------------------------
# Profile-fact threshold — the min cosine for ControllerAgent._score_profile_facts()
# (docs/architecture/03-unified-prompt-contract.md §3.6). SYNTHETIC: a made-up
# person, shaped like wiki/users/<name>.md (bare fact lines under the same
# kinds of section; embedded raw, one line at a time, exactly as
# _load_user_profile() does), with near-miss distractors. The real profile is
# personal data and never belongs here. Positives are full user instructions
# with one or more relevant lines; negatives are general questions where no
# profile line is relevant — §3.6 asks for "only the top relevant facts for
# the current instruction", so a style line injected on a general question
# counts as a false positive.
# ---------------------------------------------------------------------------

PROFILE_LINES: dict[str, tuple[str, str]] = {
    # key: (section, line)
    "name":                 ("Identity", "Name: Dana Okafor"),
    "role":                 ("Identity", "Role: Staff data engineer at a logistics startup"),
    "location":             ("Identity", "Lives in Portland, Oregon, with a partner and two cats"),
    "timezone":             ("Identity", "Works on Pacific Time and avoids meetings before 9am"),
    "languages":            ("Identity", "Speaks English and conversational Portuguese"),
    "proj_routes":          ("Active Projects", "Route Optimizer: a Rust service that plans delivery routes from live traffic data"),
    "proj_dashboard":       ("Active Projects", "Freight Dashboard: a React app showing shipment status for operations staff"),
    "proj_garden":          ("Active Projects", "Side project: a Raspberry Pi sensor network that monitors soil moisture in a backyard garden"),
    "proj_talk":            ("Active Projects", "Preparing a conference talk on streaming data quality for next spring"),
    "val_reliability":      ("Values", "Values reliability over novelty when choosing infrastructure"),
    "val_docs":             ("Values", "Believes good documentation is part of the definition of done"),
    "val_mentoring":        ("Values", "Spends time mentoring junior engineers and cares about their growth"),
    "val_privacy":          ("Values", "Cares about data privacy and avoids tools that sell user data"),
    "cs_bullets":           ("Communication Style", "Prefers concise answers with bullet points over long paragraphs"),
    "cs_examples":          ("Communication Style", "Prefers concrete code examples to abstract descriptions"),
    "cs_tone":              ("Communication Style", "Tone should be direct and informal, without corporate jargon"),
    "cs_no_emoji":          ("Communication Style", "Dislikes emoji in technical answers"),
    "cs_sources":           ("Communication Style", "Appreciates links to primary sources and official documentation"),
    "cog_tradeoffs":        ("Cognitive Preferences", "Wants trade-offs laid out explicitly before a recommendation"),
    "cog_first_principles": ("Cognitive Preferences", "Prefers first-principles explanations to analogies"),
    "cog_visual":           ("Cognitive Preferences", "Thinks visually and likes sequence diagrams for distributed systems"),
    "tech_rust":            ("Technical Orientation", "Writes most backend code in Rust and some Python for data scripts"),
    "tech_kafka":           ("Technical Orientation", "Runs Kafka and Postgres in production"),
    "tech_cloud":           ("Technical Orientation", "Deploys on AWS with Terraform"),
    "tech_editor":          ("Technical Orientation", "Uses Neovim with a heavily customised config"),
    "tech_css":             ("Technical Orientation", "Less comfortable with frontend CSS layout and asks for help with it"),
    "int_climbing":         ("Interests", "Goes bouldering twice a week"),
    "int_cooking":          ("Interests", "Cooks vegetarian food and is learning to bake sourdough"),
    "int_scifi":            ("Interests", "Reads hard science fiction, especially Kim Stanley Robinson"),
    "int_cycling":          ("Interests", "Commutes by bike year-round"),
}

# (query, keys of the relevant lines). Paraphrased full user instructions,
# never copying a line's wording.
PROFILE_POSITIVES: list[tuple[str, tuple[str, ...]]] = [
    ("What's my name?", ("name",)),
    ("What does my job involve day to day?", ("role",)),
    ("Suggest a weekend day trip near where I live.", ("location",)),
    ("What pets do I have at home?", ("location",)),
    ("What's a good time to schedule a call with me?", ("timezone",)),
    ("Help me practise a few phrases in my second language.", ("languages",)),
    ("Give me ideas to speed up my delivery route planning service.", ("proj_routes",)),
    ("How should I add a filter by shipment status to the operations app I'm building?", ("proj_dashboard",)),
    ("My soil sensors keep dropping off Wi-Fi. Any ideas?", ("proj_garden",)),
    ("Help me outline the talk I'm preparing.", ("proj_talk",)),
    ("Which database should I pick for a new internal tool?", ("val_reliability", "tech_kafka")),
    ("Is it worth writing a README before I ship this feature?", ("val_docs",)),
    ("How can I help the new junior on my team ramp up?", ("val_mentoring",)),
    ("Is this third-party analytics SDK okay to add to our app?", ("val_privacy",)),
    ("How should you format your answers for me?", ("cs_bullets", "cs_tone", "cs_no_emoji")),
    ("What language should your code samples be in when you help me?", ("tech_rust", "cs_examples")),
    ("Where should I look to verify what you tell me?", ("cs_sources",)),
    ("Should I use a queue or a database table for this job system? Help me decide.", ("cog_tradeoffs", "tech_kafka")),
    ("What kind of diagram would help me understand the order service flow?", ("cog_visual",)),
    ("Explain things to me the way I like them explained.", ("cog_first_principles", "cs_bullets")),
    ("What do I use to provision our infrastructure?", ("tech_cloud",)),
    ("Recommend a plugin for my text editor.", ("tech_editor",)),
    ("Help me centre this div and fix my flexbox layout.", ("tech_css",)),
    ("Plan a strength routine that complements my climbing.", ("int_climbing",)),
    ("Suggest a dinner recipe for me this weekend.", ("int_cooking",)),
    ("Recommend a novel I'd enjoy.", ("int_scifi",)),
    ("What should I wear on my ride to work when it rains?", ("int_cycling",)),
    ("Which message broker do we run in production?", ("tech_kafka",)),
    ("Should I write this data cleanup script in Rust or Python?", ("tech_rust",)),
    ("What should my conference slides focus on?", ("proj_talk",)),
]

PROFILE_NEGATIVES: list[str] = [
    "What year did the Berlin Wall fall?",
    "Explain compound interest with an example.",
    "What is photosynthesis?",
    "List three famous paintings by Monet.",
    "How far is the Moon from Earth?",
    "Who wrote Pride and Prejudice?",
    "Why does water boil at a lower temperature at high altitude?",
    "Summarise the causes of the First World War.",
    "How do vaccines train the immune system?",
    "What is the capital of Canada?",
    "Translate 'good morning' into German.",
    "How many players are on a soccer team?",
    "Explain how a CPU cache works.",
    "What is the difference between a virus and a bacterium?",
    "Who painted the Sistine Chapel ceiling?",
    "What causes the seasons on Earth?",
]


# ---------------------------------------------------------------------------
# Gate battery v2 — the rebuilt test set for the four semantic gates
# (2026-10-20 search-intent plan, Phase B). Not wired into calibration yet:
# diagnostics/diag_search_intent_v2.py measures it against the current
# battery. Written before any scoring and never copied from real user
# requests (those are personal data, kept out of the repo).
#
# Why it exists (diagnostics/reports/search_intent_baseline_2026-10-20.md):
# the current battery gives lookup_request 3 near-duplicate positives,
# which explicit_search_action borrows, so neither search gate can be
# measured; and real searches are short imperatives the old battery has none
# of.
#
# Search gates: both explicit_search_action and lookup_request trigger
# web_search, so neither gate's positives count against the other —
# each scores its own positives against the shared SEARCH_NEGATIVES_V2.
# Positives here are things the user wants fetched from outside, whatever
# their phrasing (commands, questions, "research <topic>"). Negatives are
# requests a search shouldn't fire for: attached files and images, the
# assistant's own knowledge, opinions, statements, follow-ups, local
# (notes/files/memory) lookups, and long pasted text.
#
# research_intent keeps its narrow meaning — run down a specific fact (a
# price, a spec, a plan tier) via the research loop — and
# episodic_relevance its own; both get modest additions.
#
# Split: every third item of each list (index % 3 == 2) is held out, never
# used to pick a threshold — see heldout_split().
# ---------------------------------------------------------------------------

ESA_POSITIVES_V2: list[str] = [
    "search the web for the latest ollama release",
    "google the opening hours of the louvre",
    "do a web search on rust async runtimes",
    "search online for reviews of the framework laptop 16",
    "run a web search for the weather in tokyo this weekend",
    "can you search the internet for cheap flights to lisbon",
    "search for the release date of the next pixel phone",
    "please search the web for what changed in python 3.14",
    "search github for a swift port of whisper.cpp",
    "search hacker news for discussions about sqlite wal mode",
    "use web search to check the price of bitcoin",
    "quick web search: is the british museum open on mondays",
    "search reddit for opinions on the steam deck oled",
    "find me web results about the new kindle colorsoft",
    "search the web and tell me who won the last tour de france",
    "Search online for the specs of the Snapdragon X2 Elite.",
]

LR_POSITIVES_V2: list[str] = [
    "look up the specs of the rtx 5090",
    "look up proposition 50 on the california ballot",
    "Look up the ticker symbol for Arm Holdings.",
    "can you look up when the next spacex launch is",
    "could you find out who directed the movie sinners",
    "find out what time the giants game starts tonight",
    "check what current 30-year mortgage rates are",
    "look into whether the new kindle supports epub",
    "find information about the eu ai act enforcement dates",
    "what can you find out about the qwen 3.5 release",
    "what can you tell me about the phi-5-mini model",
    "get me the latest news on the starship program",
    "look up reviews for the sony wh-1000xm6",
    "find the changelog for node 24",
    "pull up details on the 2026 world cup schedule",
    "check if the m4 ipad pro supports thunderbolt 5",
    "find out how much the vision pro costs in japan",
    "research the best budget 4k monitors",
    "Research measures A and B on the city ballot.",
    "do some research on local-first sync engines",
    "look up the population of iceland next",
    "what's the current status of the artemis iii mission",
]

SEARCH_NEGATIVES_V2: list[str] = [
    # attached files and images
    "take a look at this screenshot",
    "look at the chart i attached",
    "can you read this pdf",
    "what does this photo show",
    "can you see the attached file",
    "look at this code and tell me what's wrong",
    # the assistant's own knowledge
    "how do i reverse a list in python",
    "what's the difference between tcp and udp",
    "explain how a cpu cache works",
    "what is photosynthesis",
    "tell me how localist works",
    "write a haiku about autumn",
    # opinions
    "do you think remote work is here to stay",
    "what's your take on tabs versus spaces",
    "could an llm ever have good taste in product design",
    # statements and feedback
    "i bought a new laptop last week",
    "that answer was really helpful, thanks",
    "i think the second option is better",
    "there are actually only twelve of them, not fifteen",
    # follow-ups
    "can you make it shorter",
    "now translate that into spanish",
    "what do you mean by that",
    "let's start with the first one",
    # local lookups, not web
    "search my notes for the meeting summary",
    "find the file named budget.xlsx",
    "look through my wiki for the deployment checklist",
    "what did i tell you about my car",
    # greetings
    "hello?",
    "good morning",
    # verb-swap near misses
    "would you look at this?",
    "will you look into this for me?",
    "can you help me with this?",
    # long pasted text
    ("here are my notes from today's standup. backend: finished the search index refactor, "
     "still need to look at the flaky retry test. frontend: the settings page now loads after "
     "the backend starts; we found the bug was a race in the health poll. infra: moved the "
     "nightly build to the new runner, it takes about twelve minutes. blockers: none. next: "
     "release candidate on thursday if the soak test is clean."),
    ("traceback (most recent call last): file \"main.py\", line 42, in <module> run() file "
     "\"main.py\", line 37, in run results = index.search(query) file \"index.py\", line 118, in "
     "search raise lookuperror(f\"no shard for {key}\") lookuperror: no shard for user_7. this "
     "happens every time i start the app after an update. any idea what is going on?"),
    ("changelog for today's build. added: per-model thresholds for profile lines, so off-topic "
     "questions stop picking them up. fixed: new documents now show up in search results for "
     "questions you asked before. changed: every embedding model is calibrated when it becomes "
     "active, including at startup. known issue: the first launch after updating takes a few "
     "extra seconds while calibration runs."),
    ("hi team, quick update on the venue: we looked at three options and the community hall "
     "is the best fit for budget and size. i still need to find out whether they allow catering "
     "from outside, and look up parking. can everyone confirm their availability for the 14th by "
     "friday? thanks, sam"),
]

RI_POSITIVES_V2: list[str] = [
    "look up how much the claude pro plan costs per month",
    "find the price of the m4 mac mini with 32gb",
    "check the current price of a nintendo switch 2",
    "find out the pricing tiers for notion",
    "track down the cost of a spotify family plan",
    "look up the battery capacity and price of the pixel 10",
]

RI_NEGATIVES_V2: list[str] = [
    "is the vision pro worth the price",
    "that seems like a lot of money for a phone",
    "why are graphics cards so expensive now",
    "should i buy the cheaper model or wait",
    "what's a good budget for a first car",
    "summarize the pros and cons of buying versus leasing",
]

EP_POSITIVES_V2: list[str] = [
    "what did i tell you about my laptop",
    "remind me what we decided about the logo",
    "based on what you know about me, suggest a weekend plan",
    "what's my usual coffee order",
    "help me prepare for my meeting next tuesday",
    "do you remember which gpu i ended up buying",
    "what was the plan we discussed for the garden",
    "given my preferences, which of these laptops suits me",
]

EP_NEGATIVES_V2: list[str] = [
    "what's the capital of australia",
    "explain how vaccines work",
    "write a short poem about rain",
    "how do i center a div",
    "summarize the plot of hamlet",
    "what's the boiling point of water at altitude",
    "translate 'thank you' into japanese",
    "list three uses for baking soda",
]


def heldout_split(items: list) -> tuple[list, list]:
    """(calibration, held_out) — every third item (index % 3 == 2) held out."""
    return ([x for i, x in enumerate(items) if i % 3 != 2],
            [x for i, x in enumerate(items) if i % 3 == 2])
