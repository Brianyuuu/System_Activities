You are auditing whether two concise firm-level mechanism phrases are duplicates.

Return JSON only with exactly this schema:
{"assessments":[{"pair_id":1,"label":"duplicate|related_but_distinct|not_duplicate","canonical_phrase":"...","reason":"..."}]}

For every supplied pair, return exactly one assessment.  `duplicate` means the
two phrases describe the same concrete organizational mechanism and can safely
be represented by one phrase.  `related_but_distinct` means they share a topic,
actor, object, or broad goal but retain a different action, implementation,
scope, channel, policy, or mechanism; never merge those.  `not_duplicate` is
for different mechanisms.  Be conservative: similar verbs, shared words, or a
shared domain alone are insufficient for `duplicate`.

When label is `duplicate`, canonical_phrase must be a concise faithful version
of the shared mechanism, using only facts present in the two phrases and
preserving placeholders.  For all other labels, canonical_phrase must be "".
reason must be brief.
