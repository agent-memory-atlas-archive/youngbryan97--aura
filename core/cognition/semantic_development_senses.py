"""How a word's senses are read from the contexts it was used in.

Lifted whole out of `semantic_development`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from core.cognition.serialized_method import serialized as _serialized

if TYPE_CHECKING:
    from .semantic_development import (
        SemanticCase,
        UsageEvent,
    )


class _ReadsSensesInContext:
    """Lifted whole out of SemanticDevelopment; see semantic_development.py."""

    def _sense_evidence(self, term: str, *, as_of: float | None = None,
                        ) -> dict[tuple[str, str], bool | None]:
        from .semantic_development import (
            defaultdict,
        )

        stances: dict[tuple[str, str], set[str]] = defaultdict(set)
        for feedback in self.meaning_feedback.values():
            if (feedback.term.casefold() == term
                    and (as_of is None or feedback.observed_at <= as_of)
                    and feedback.usage_source_id in self.usage_events
                    and (as_of is None or self.usage_events[
                        feedback.usage_source_id].observed_at <= as_of)):
                stances[(feedback.usage_source_id, feedback.sense)].add(feedback.stance)
        return {key: (None if len(values) != 1 else "supports" in values)
                for key, values in stances.items() if key[0] in self.usage_events}

    def _labeled_usage(self, term: str, *, as_of: float | None = None,
                       ) -> tuple[tuple[UsageEvent, str], ...]:

        return tuple((self.usage_events[source], sense)
                     for (source, sense), supported in self._sense_evidence(
                         term, as_of=as_of).items()
                     if supported is True)

    @staticmethod
    def _feature_measured(event: UsageEvent, feature: str, value: Any,
                          features: Mapping[str, Any]) -> bool:

        if feature.startswith(("cue:", "context:", "relation:", "pragmatic:")):
            return feature in features
        if event.original_token_count > len(event.terms):
            if feature.startswith("co:"):
                return feature in features
            if feature in {"usage:spoken", "usage:frequency"}:
                return bool(value)
        return True

    @_serialized
    def usage_associations(
        self, term: str, *, setting: str = "", community: str = "", limit: int = 16,
    ) -> dict[str, Any]:
        """Compare observed co-use with its local base rate, not taxonomy."""
        from .semantic_development import (
            Counter,
        )

        if not 1 <= limit <= 64:
            raise ValueError("association limit is out of range")
        key = term.casefold()
        scoped = [event for event in self.usage_events.values()
                  if (not setting or event.setting == setting)
                  and (not community or event.community == community)]
        exposed = [event for event in scoped
                   if key in event.terms or key in event.referents]
        if not exposed:
            status = ("unmeasured_due_to_sampling" if any(
                event.original_token_count > len(event.terms) for event in scoped)
                else "unexposed")
            return {"status": status, "term": key, "observed_sources": 0,
                    "associations": (), "serving_authority": False}
        background = Counter(neighbor for event in scoped
                             for neighbor in set(event.terms))
        neighbors = Counter(neighbor for event in exposed
                            for neighbor in set(event.terms) - {key})
        associations = []
        for neighbor, support in neighbors.items():
            base = background[neighbor] / len(scoped)
            conditional = support / len(exposed)
            associations.append({"term": neighbor, "sources": support,
                                 "conditional_frequency": conditional,
                                 "background_frequency": base,
                                 "lift": conditional / base if base else 0.0,
                                 "relation": "observed_co_use"})
        associations.sort(key=lambda row: (-row["sources"], -row["lift"], row["term"]))
        cues = Counter((cue.channel, cue.name, str(cue.value))
                       for event in exposed for cue in event.cues)
        stretched = sum(key in event.stretched_terms for event in exposed)
        indirect = sum(key in event.referents and key not in event.terms
                       for event in exposed)
        return {"status": "observed_association", "term": key,
                "observed_sources": len(exposed),
                "partially_sampled_sources": sum(
                    event.original_token_count > len(event.terms) for event in exposed),
                "associations": tuple(associations[:limit]),
                "delivery_cues": tuple({"channel": channel, "name": name,
                                         "value": value, "sources": count}
                                        for (channel, name, value), count in cues.most_common(limit)),
                "stretched_sources": stretched, "indirect_sources": indirect,
                "serving_authority": False}

    @_serialized
    def contextual_senses(self, term: str, situation: UsageEvent, *,
                          excluded_sources: tuple[str, ...] = (),
                          as_of: float | None = None) -> dict[str, Any]:
        """Rank grounded readings by similar prior use; absence stays absence."""

        return self._contextual_labels(term, situation,
                                       self._labeled_usage(term.casefold(), as_of=as_of),
                                       self._sense_evidence(term.casefold(), as_of=as_of),
                                       excluded_sources=excluded_sources,
                                       label_name="sense")

    def _contextual_labels(
        self, term: str, situation: UsageEvent,
        labeled_usage: Sequence[tuple[UsageEvent, str]],
        explicit_refutations: Mapping[tuple[str, str], bool | None], *,
        excluded_sources: tuple[str, ...], label_name: str,
        query_extra: Mapping[str, Any] | None = None,
        source_extra: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        from .semantic_development import (
            defaultdict,
        )

        key = term.casefold()
        if key not in situation.terms and key not in situation.referents:
            raise ValueError("the situation does not contain the concept")
        excluded = {*excluded_sources, situation.source_id}
        labeled = [(event, sense) for event, sense in labeled_usage
                   if event.source_id not in excluded]
        if not labeled:
            return {"status": "unexposed_to_grounded_sense", "term": key,
                    "candidates": (), "serving_authority": False}
        query = {**situation.features_for(key), **(query_extra or {})}
        grouped: dict[str, list[UsageEvent]] = defaultdict(list)
        for event, sense in labeled:
            grouped[sense].append(event)
        features_by_source = {
            event.source_id: {**event.features_for(key),
                              **(source_extra or {}).get(event.source_id, {})}
            for event, _sense in labeled
        }
        query_features = frozenset(
            (name, value) for name, value in query.items()
            if name != "usage:sampled"
            and self._feature_measured(situation, name, value, query))
        local = len({event.source_id for event, _sense in labeled
                     if (not situation.setting or event.setting == situation.setting)
                     and (not situation.community or event.community == situation.community)})
        labeled_sources = {event.source_id for event, _sense in labeled}
        candidates = []
        for sense, examples in grouped.items():
            positive_sources = {event.source_id for event in examples}
            rival_sources = {event.source_id: event for event, _label in labeled
                             if event.source_id not in positive_sources}
            rival_sources.update({source: self.usage_events[source]
                                  for (source, reading), supported in explicit_refutations.items()
                                  if reading == sense and supported is False
                                  and source not in positive_sources
                                  and source not in excluded})
            negatives = list(rival_sources.values())
            score = math.log((len(examples) + 1) / (len(labeled_sources) + len(grouped)))
            discriminators = []
            for feature in query_features:
                name, value = feature
                measured_positive = [features_by_source[event.source_id]
                                     for event in examples
                                     if self._feature_measured(
                                         event, name, value,
                                         features_by_source[event.source_id])]
                measured_negative = []
                for event in negatives:
                    features = features_by_source.get(event.source_id)
                    if features is None:
                        features = event.features_for(key)
                    if self._feature_measured(event, name, value, features):
                        measured_negative.append(features)
                if not measured_positive or not measured_negative:
                    continue
                positive = sum(item.get(name) == value for item in measured_positive)
                negative = sum(item.get(name) == value for item in measured_negative)
                contribution = math.log(
                    ((positive + 1) / (len(measured_positive) + 2)) /
                    ((negative + 1) / (len(measured_negative) + 2)))
                score += contribution
                if contribution > 0:
                    discriminators.append((feature[0], contribution))
            candidates.append({label_name: sense,
                               "independent_sources": len({event.source_id for event in examples}),
                               "evidence_score": score,
                               "supporting_features": tuple(name for name, _ in sorted(
                                   discriminators, key=lambda item: (-item[1], item[0]))[:8]),
                               "relation": "contrastive_abductive_context_fit"})
        candidates.sort(key=lambda row: (-row["evidence_score"],
                                         -row["independent_sources"], row[label_name]))
        tied = (len(candidates) > 1 and math.isclose(
            candidates[0]["evidence_score"], candidates[1]["evidence_score"],
            abs_tol=1e-9))
        status = ("unmeasured_context_transfer" if not local else
                  "no_discriminating_evidence" if tied else "ranked_hypotheses")
        return {"status": status,
                "term": key, "local_grounded_sources": local,
                "candidates": tuple(candidates), "serving_authority": False}

    def _context_relation_fit(self, event: UsageEvent, *, as_of: float,
                              excluded_sources: tuple[str, ...]) -> tuple[dict[str, Any],
                                                                           dict[str, Any]]:
        from .semantic_development import (
            compare_pragmatic_context,
        )

        context = sorted((other for other in self.usage_events.values()
                          if other.context_id == event.context_id
                          and other.source_id not in (*excluded_sources, event.source_id)
                          and other.observed_at <= as_of),
                         key=lambda other: (abs(other.observed_at - event.observed_at),
                                            other.source_id))[:32]
        result = compare_pragmatic_context(event, context, as_of=as_of)
        if not result["comparable"]:
            return {}, result
        kind = ("mixed" if result["aligned"] and result["incongruent"] else
                "incongruent" if result["incongruent"] else "aligned")
        return {"pragmatic:relation_fit": kind}, result

    @_serialized
    def pragmatic_readings(
        self, term: str, situation: UsageEvent, *,
        excluded_sources: tuple[str, ...] = (), as_of: float | None = None,
    ) -> dict[str, Any]:
        """Compare scoped claims and rank attributed indirect readings.

        A mismatch is not a motive. Only source-attributed feedback can teach
        metaphor, fiction, joke, mistake, or deception; unresolved feedback
        is never converted into an intent label.
        """

        key = term.casefold()
        if key not in situation.terms and key not in situation.referents:
            raise ValueError("the situation does not contain the concept")
        cutoff = situation.observed_at if as_of is None else float(as_of)
        if not math.isfinite(cutoff) or cutoff < situation.observed_at:
            raise ValueError("pragmatic cutoff must follow the focal usage")
        query_extra, congruence = self._context_relation_fit(
            situation, as_of=cutoff, excluded_sources=excluded_sources)
        senses = self.contextual_senses(key, situation,
                                        excluded_sources=excluded_sources, as_of=cutoff)
        supported = self._sense_evidence(key, as_of=cutoff)
        labeled = tuple((self.usage_events[item.usage_source_id],
                         item.sense + "\0" + item.mode)
                        for item in self.meaning_feedback.values()
                        if (item.term.casefold() == key and item.stance == "supports"
                            and item.mode != "unresolved"
                            and item.observed_at <= cutoff
                            and supported.get((item.usage_source_id, item.sense)) is True
                            and item.usage_source_id in self.usage_events))
        feedback_cutoffs = {source: max(item.observed_at for item in self.meaning_feedback.values()
                                        if item.usage_source_id == source
                                        and item.observed_at <= cutoff)
                            for source in {event.source_id for event, _label in labeled}}
        source_extra = {source: self._context_relation_fit(
            self.usage_events[source], as_of=at,
            excluded_sources=(*excluded_sources, situation.source_id))[0]
            for source, at in feedback_cutoffs.items()}
        modes = self._contextual_labels(
            key, situation, labeled, {}, excluded_sources=excluded_sources,
            label_name="interpretation", query_extra=query_extra,
            source_extra=source_extra)
        candidates = tuple({**row, "sense": row["interpretation"].split("\0", 1)[0],
                            "mode": row["interpretation"].split("\0", 1)[1]}
                           for row in modes["candidates"])
        indirect = tuple(ref for ref in situation.referents if ref not in situation.terms)
        discriminated = (len(candidates) > 1 and bool(candidates[0]["supporting_features"])
                         and modes["status"] == "ranked_hypotheses")
        return {"status": "ranked_hypotheses" if discriminated
                else "unresolved", "term": key, "source_id": situation.source_id,
                "as_of": cutoff, "congruence": congruence,
                "indirect_referents": indirect,
                "sense_hypotheses": senses["candidates"],
                "interpretation_hypotheses": candidates,
                "intent": "unmeasured", "serving_authority": False}

    @_serialized
    def evaluate_contextual_senses(self, term: str,
                                   heldout_sources: tuple[str, ...]) -> dict[str, Any]:
        """Measure interpretations with every held-out source excluded from fitting."""
        from .semantic_development import (
            Counter,
            defaultdict,
        )

        if not 1 <= len(heldout_sources) <= 256 or len(set(heldout_sources)) != len(
                heldout_sources):
            raise ValueError("sense evaluation needs distinct bounded held-out sources")
        labeled: dict[str, tuple[UsageEvent, set[str]]] = {}
        for event, sense in self._labeled_usage(term.casefold()):
            labeled.setdefault(event.source_id, (event, set()))[1].add(sense)
        if any(source not in labeled for source in heldout_sources):
            raise ValueError("held-out source lacks unambiguous attributed feedback")
        training_labels = Counter(sense for source, (_event, senses) in labeled.items()
                                  if source not in heldout_sources for sense in senses)
        if not training_labels:
            return {"status": "unmeasured_no_training_exposure", "n": 0,
                    "serving_authority": False}
        majority = max(training_labels, key=lambda sense: (training_labels[sense], sense))
        correct = baseline_correct = answered = single_label_n = single_label_correct = 0
        by_community: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for source in heldout_sources:
            event, senses = labeled[source]
            result = self.contextual_senses(
                term, event, excluded_sources=heldout_sources)
            predicted = (result["candidates"][0]["sense"]
                         if result["status"] == "ranked_hypotheses" else None)
            answer = int(predicted is not None)
            hit = int(predicted in senses)
            answered += answer
            correct += hit
            baseline_correct += int(majority in senses)
            if len(senses) == 1:
                single_label_n += 1
                single_label_correct += hit
            row = by_community[event.community or "unspecified"]
            row[0] += 1
            row[1] += answer
            row[2] += hit
        return {"status": "measured_development_only", "n": len(heldout_sources),
                "answered": answered, "correct": correct,
                "correct_any_supported_sense": correct,
                "unambiguous_n": single_label_n,
                "unambiguous_correct": single_label_correct,
                "majority_baseline_correct": baseline_correct,
                "heldout_sources": heldout_sources,
                "by_community": {name: {"n": counts[0], "answered": counts[1],
                                        "correct": counts[2]}
                                 for name, counts in sorted(by_community.items())},
                "serving_authority": False}

    @_serialized
    def discriminating_usage_observations(self, term: str, situation: UsageEvent,
                                          *, limit: int = 8) -> dict[str, Any]:
        """Identify measured distinctions worth checking in an unresolved setting."""
        from .semantic_development import (
            Counter,
            defaultdict,
        )

        if not 1 <= limit <= 32:
            raise ValueError("discriminating observation limit is out of range")
        key = term.casefold()
        labeled = [(event, sense) for event, sense in self._labeled_usage(key)
                   if event.source_id != situation.source_id]
        if len({sense for _event, sense in labeled}) < 2:
            return {"status": "insufficient_rivals", "observations": (),
                    "serving_authority": False}
        grouped: dict[str, list[UsageEvent]] = defaultdict(list)
        for event, sense in labeled:
            grouped[sense].append(event)
        observed_names = set(situation.features_for(key))
        candidates: dict[tuple[str, Any], dict[str, float]] = defaultdict(dict)
        for sense, examples in grouped.items():
            feature_sets = [set(event.features_for(key).items()) for event in examples]
            frequencies = Counter(feature for features in feature_sets for feature in features)
            for feature, count in frequencies.items():
                candidates[feature][sense] = count / len(examples)
        ranked = []
        for (name, value), rates in candidates.items():
            if (name in observed_names or name.startswith("usage:")
                    or name == "context:speaker"):
                continue
            if name.startswith(("cue:", "context:")) and any(
                any(name not in event.features_for(key) for event in examples)
                for examples in grouped.values()
            ):
                continue
            if name.startswith("co:") and any(
                event.original_token_count > len(event.terms)
                for examples in grouped.values() for event in examples
            ):
                continue
            complete = {sense: rates.get(sense, 0.0) for sense in grouped}
            separation = max(complete.values()) - min(complete.values())
            if separation <= 0:
                continue
            ranked.append({"feature": name, "value": value,
                           "separation": separation,
                           "observed_rates": complete,
                           "status": "proposed_observation_not_evidence"})
        ranked.sort(key=lambda row: (-row["separation"], row["feature"], str(row["value"])))
        return {"status": "candidate_observations" if ranked else
                "no_discriminating_observation", "term": key,
                "observations": tuple(ranked[:limit]),
                "serving_authority": False}

    @_serialized
    def associative_analogy(self, left: str, right: str, *, setting: str = "",
                            community: str = "") -> dict[str, Any]:
        """Compare use-neighborhoods while keeping kind-of claims separate."""
        first = self.usage_associations(left, setting=setting, community=community)
        second = self.usage_associations(right, setting=setting, community=community)
        if (first["status"] != "observed_association" or
                second["status"] != "observed_association"):
            return {"status": "unmeasured" if "unmeasured_due_to_sampling" in (
                first["status"], second["status"]) else "unexposed",
                "shared": (), "serving_authority": False}
        left_neighbors = {row["term"] for row in first["associations"]}
        right_neighbors = {row["term"] for row in second["associations"]}
        shared = left_neighbors & right_neighbors
        union = left_neighbors | right_neighbors
        return {"status": "association_analogy", "left": left.casefold(),
                "right": right.casefold(), "shared": tuple(sorted(shared)),
                "jaccard": len(shared) / len(union) if union else 0.0,
                "relation": "shared_observed_use_not_taxonomy",
                "serving_authority": False}

    @_serialized
    def grounded_sense_cases(self, term: str, sense: str) -> tuple[SemanticCase, ...]:
        """Expose feedback as testable cases, not as an admitted definition."""
        from .semantic_development import (
            SemanticCase,
        )

        key = term.casefold()
        return tuple(SemanticCase(
            event.source_id, event.context_id, f"usage_sense:{key}:{sense}",
            supported, event.features_for(key), event.observed_at,
            intervention="meaning_feedback")
            for (source, reading), supported in self._sense_evidence(key).items()
            if reading == sense and supported is not None
            for event in (self.usage_events[source],))

    @_serialized
    def relation_evidence(self, left: str, right: str, *, atomspace: Any = None) -> dict[str, Any]:
        """Report story association apart from an explicitly stored kind-of edge."""
        from core.knowledge.atomspace import INHERITANCE, Link, concept, get_atomspace

        if atomspace is None:
            atomspace = get_atomspace()
        inheritance = atomspace.get_tv(Link(INHERITANCE, (concept(left), concept(right))))
        co_used = sum(left.casefold() in event.terms and right.casefold() in event.terms
                      for event in self.usage_events.values())
        return {"left": left.casefold(), "right": right.casefold(),
                "co_use_sources": co_used,
                "co_use_relation": "narrative_or_situational_association" if co_used
                else "unmeasured",
                "taxonomic_relation": ({"strength": inheritance.strength,
                                        "confidence": inheritance.confidence}
                                       if inheritance is not None else None),
                "serving_authority": False}

