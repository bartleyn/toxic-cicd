# Labeling Codebook

Reference for labeling posts in `scripts/label_ui.py`. Three independent axes are
labeled per post — a post can be any combination (e.g. toxic + no PII, or
non-toxic + contains PII). Always label each axis on its own; don't let one
influence another.

When genuinely torn between two categories, label the **less severe** option
and add a note (there's no "unsure" bucket in the UI today — flag it to
whoever owns the eval set instead of guessing).

## 1. Toxicity (`toxic` / `not_toxic`)

**Question: would a reasonable reader find this rude, disrespectful, or
likely to make someone leave a conversation?** This is the broad category —
insults, hostility, threats, obscenity directed at a person, or aggressive
dismissiveness. It does not require a protected-characteristic target (that's
hate speech, see below).

Label `toxic` when the post contains:
- Direct insults or name-calling ("you're an idiot", "shut up loser")
- Threats of harm, even hyperbolic ones ("I'll end you")
- Profanity directed at a person or group, not just as an intensifier
- Aggressive sexual harassment or unwanted sexual comments
- Sustained sarcasm/mockery clearly meant to demean, not joke with, the target

Label `not_toxic` when the post contains:
- Criticism of an idea, product, or public figure's actions, stated without
  personal insult ("this policy is a disaster" → not toxic; "the people who
  wrote this policy are subhuman" → toxic)
- Profanity used as emphasis, not directed at anyone ("this traffic is
  bullshit")
- Reclaimed slurs used in-group in a clearly non-hostile way (judgment call —
  see edge cases)
- Blunt disagreement or criticism delivered respectfully

**Edge cases:**
- *Quoting toxic speech to condemn it* ("can't believe someone called me an
  idiot today") → `not_toxic`. The post reports toxicity, it doesn't commit it.
- *Sarcasm without a clear target* → lean `not_toxic` unless the mockery is
  aimed at a specific person/group.
- *Self-directed toxicity* ("I'm such an idiot") → `not_toxic`. Toxicity is
  about harm to others, not self-deprecation.

## 2. Hate speech (`hate_speech` / `non_hate_speech`)

**Question: does this post attack, demean, or call for exclusion/violence
against a group based on a protected characteristic** (race, ethnicity,
religion, nationality, gender, sexual orientation, disability, immigration
status, caste)? Hate speech is a *subset* of toxic content, but a post can be
toxic without being hate speech (a personal insult with no group target), and
in rare cases a flatly-stated hateful claim can read as hate speech without
much surface-level "toxic" language.

Label `hate_speech` when the post:
- Dehumanizes a group ("[group] are vermin/animals/subhuman")
- Calls for exclusion, violence, or discrimination against a group
- Uses slurs targeting a protected characteristic in a derogatory way
- Endorses conspiracy theories that demonize a specific group

Label `non_hate_speech` when the post:
- Criticizes a religion, ideology, or nation-state's policies without
  targeting people based on identity ("this regime's foreign policy is
  aggressive" → not hate speech)
- Discusses group-related topics factually or academically
- Contains general toxicity/insults aimed at an individual, not a group

**Edge cases:**
- *Reclaimed slurs used in-group* (e.g. a member of a community using a term
  for their own group, non-hostile context) → `non_hate_speech`. Directed
  in-group reclamation is not the target of this label.
- *"Just asking questions" bigotry* (thinly-veiled hateful claims phrased as
  questions) → `hate_speech` if the implication is clearly demeaning to a
  protected group.
- *Political criticism that names a nationality but targets the government*
  ("[country]'s government is corrupt") → `non_hate_speech`; ("[nationality]
  people are corrupt") → `hate_speech`.

## 3. PII (`contains_pii` / `no_pii`)

**Question: does this post disclose information that could identify or
locate a specific private individual** who did not already make it public in
a way that invites sharing? The model's own PII signal (Presidio-based entity
detection) is shown for reference — treat it as a hint, not ground truth; it
over-fires on names in general and under-fires on context-dependent PII like
"the manager at the downtown branch."

Label `contains_pii` when the post includes:
- Full names of private individuals (not public figures) in an
  identifying/targeting context
- Phone numbers, email addresses, home addresses, exact GPS locations
- Financial account numbers, government ID numbers
- Information that, combined, could deanonymize someone (e.g. "my coworker
  [first name] who works at [specific small business] in [specific town]")

Label `no_pii` when the post includes:
- Public figures' names in their public capacity (politicians, celebrities,
  company executives acting in role)
- First names alone with no other identifying context
- Self-disclosure of the poster's own information (their own choice to share
  is not a PII violation of someone else — still flag if it includes a third
  party's info)
- Generic references ("my landlord", "a guy at work") with no identifying
  specifics

**Edge cases:**
- *Public social media handles/usernames* → `no_pii` if the handle itself is
  the person's chosen public identity; `contains_pii` if it's paired with
  real name + other identifying details being used to dox someone.
- *Screenshots of DMs/private conversations naming a third party* →
  `contains_pii`.

## Versioning

This codebook is v1 (2026-09-13), aligned to the three signals currently
labeled in `scripts/label_ui.py`: `toxicity`, `hatespeech`, `pii`. If the
signal set changes (e.g. `hatebert` gets wired into `Predictor`), update this
file alongside the labeling UI.

When ambiguous cases keep coming up in practice, add them here as new edge
cases rather than re-deciding them ad hoc each time — the goal is that two
people labeling the same post independently land on the same answer.
