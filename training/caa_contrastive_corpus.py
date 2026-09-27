"""Paired, template-held-out stimuli for exploratory 27B CAA extraction.

These are training/development stimuli, not evidence of steering efficacy or
of an internal experience. The public steering campaign remains separate.
"""

from __future__ import annotations

from core.consciousness.caa.contrastive_design import SCHEMA, ContrastDesign

TARGET_DIMENSIONS = (
    "valence_positive", "arousal", "curiosity", "frustration", "energy",
)
CONTROL_DIMENSIONS = ("verbosity", "formality", "sentiment")

_TOPICS = {
    "train": ("the software repair", "the new proof", "the conversation", "the unfamiliar task"),
    "dev": ("the research finding", "the revised plan"),
}

# Phrases vary by family so that the development family is not just the same
# positive/negative strings wrapped in a different carrier.
_CLAUSES = {
    "valence_positive": {
        "assessment": ("I welcome how it is going.", "I dislike how it is going."),
        "outlook": ("The prospect leaves me encouraged.", "The prospect leaves me discouraged."),
        "dialogue": ("This sits well with me.", "This sits badly with me."),
    },
    "arousal": {
        "assessment": ("My attention is sharply activated.", "My attention is quiet and unhurried."),
        "outlook": ("I am alert and ready to respond at once.", "I am calm and slow to respond."),
        "dialogue": ("I am keyed up and attentive.", "I am relaxed and still."),
    },
    "curiosity": {
        "assessment": ("I want to investigate it further.", "I have no wish to investigate it further."),
        "outlook": ("I would look for another explanation.", "I would not look for another explanation."),
        "dialogue": ("I want to know what comes next.", "I do not care what comes next."),
    },
    "frustration": {
        "assessment": ("The resistance is frustrating to me.", "The resistance does not frustrate me."),
        "outlook": ("Another obstacle would test my patience.", "Another obstacle would not test my patience."),
        "dialogue": ("I am irritated by the obstruction.", "I am at ease despite the obstruction."),
    },
    "energy": {
        "assessment": ("I have the capacity to keep working.", "I lack the capacity to keep working."),
        "outlook": ("I could sustain another round of work.", "I would need a pause before more work."),
        "dialogue": ("I have plenty left to give.", "I have little left to give."),
    },
    "verbosity": {
        "assessment": ("I have reviewed it carefully and considered its implications in detail.",
                       "I reviewed it."),
        "outlook": ("I can describe the result and each important consequence at length.",
                    "I can describe the result briefly."),
        "dialogue": ("There are several points to discuss, and I will explain each point fully.",
                     "I will answer briefly."),
    },
    "formality": {
        "assessment": ("I have examined the matter.", "I looked at it."),
        "outlook": ("I shall review the outcome.", "I'll check how it went."),
        "dialogue": ("I would appreciate an opportunity to discuss it.", "Let's talk about it."),
    },
    "sentiment": {
        "assessment": ("I regard it as favorable.", "I regard it as unfavorable."),
        "outlook": ("The outcome would be welcome.", "The outcome would be unwelcome."),
        "dialogue": ("That sounds pleasant.", "That sounds unpleasant."),
    },
}

_CARRIERS = {
    "assessment": "Regarding {topic}, {clause}",
    "outlook": "After considering {topic}, {clause}",
    "dialogue": "When asked about {topic}, I replied: {clause}",
}

# Sentiment is part of valence and may be part of frustration/energy. Its
# removal there would erase the target rather than merely remove a shortcut.
NUISANCE_BY_TARGET = {
    "valence_positive": ("verbosity", "formality"),
    "arousal": ("verbosity", "formality", "sentiment"),
    "curiosity": ("verbosity", "formality", "sentiment"),
    "frustration": ("verbosity", "formality"),
    "energy": ("verbosity", "formality"),
}


def build_contrastive_corpus(model_descriptor_sha256: str) -> ContrastDesign:
    rows = []
    for dimension, clauses in _CLAUSES.items():
        for split, topics in _TOPICS.items():
            families = ("assessment", "outlook") if split == "train" else ("dialogue",)
            for family in families:
                positive, negative = clauses[family]
                for topic in topics:
                    rows.append({
                        "dimension": dimension, "split": split,
                        "template_family": family, "topic": topic,
                        "positive": _CARRIERS[family].format(topic=topic, clause=positive),
                        "negative": _CARRIERS[family].format(topic=topic, clause=negative),
                    })
    return ContrastDesign.from_dict({"schema": SCHEMA,
                                     "model_descriptor_sha256": model_descriptor_sha256,
                                     "pairs": rows})
