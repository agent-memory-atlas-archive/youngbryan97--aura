"""How the goal store reaches her state: the one projection of what she is working on, and the order goals are deduplicated in.

Lifted whole out of `goal_engine`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
from typing import Any


class _GoalProjectionMixin:
    """Lifted whole out of GoalEngine; see goal_engine.py."""

    def _deduplicated(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """In priority order, each goal once, and one active goal per objective and horizon."""
        from .goal_engine import (
            ACTIVE_GOAL_STATUSES,
        )

        deduped: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_active_signatures: set[str] = set()
        for item in self._sort_items(items):
            item_id = str(item.get("id", "") or f"{item.get('source','goal')}::{item.get('objective') or item.get('name')}")
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            status = str(item.get("status", "") or "")
            if status in ACTIVE_GOAL_STATUSES:
                signature = self._goal_signature(item.get("objective") or item.get("name") or "")
                horizon = str(item.get("horizon", "") or "")
                active_key = f"{horizon}::{signature}" if signature else ""
                if active_key and active_key in seen_active_signatures:
                    continue
                if active_key:
                    seen_active_signatures.add(active_key)
            deduped.append(item)

        return deduped

    def _sync_state_view(self, limit: int = 6) -> None:
        state = getattr(self.state_repo, "_current", None)
        if state is None:
            return
        cognition = getattr(state, "cognition", None)
        if cognition is None:
            return
        self.project_onto(cognition, limit=limit)

    async def _sync_state_view_off_loop(self) -> None:
        """The same projection, built on a worker thread.

        LIVE, 29 September: eight loop stalls of 5.4 to 8.1 seconds in one
        night, every one inside `_fetch_records` under `track_dispatch`,
        `update_task_lifecycle`, the task engine or initiative synthesis. The
        projection has to read what the mutation just wrote, so it builds the
        snapshot fresh, reconciliation writes included, and that is SQLite work
        with no place on the event loop. The engine's lock already serialises
        its connection across threads; the snapshot refresh relies on it.
        """
        await asyncio.to_thread(self._sync_state_view)

    def project_onto(self, cognition: Any, limit: int = 6) -> None:
        """Write the goals she is working on, and the one to get on with, onto `cognition`.

        The one owner of `cognition.active_goals`. The task engine wrote the
        raw goal list over it after every plan, which dropped the urgency the
        workspace prices deliberation's bid on, so it calls this instead.
        """
        from .goal_engine import (
            FallbackClassification,
            _origin_is_user_anchored,
            _record_goal_degradation,
            _the_one_to_get_on_with,
            is_intrinsic_goal_text,
            logger,
            what_leaving_each_costs,
        )

        try:
            active = self.get_active_goals(limit=limit, include_external=False, actionable_only=True, fresh=True)
            if not active:
                active = self.get_active_goals(limit=limit, include_external=True, actionable_only=True)
            shown = active[:limit]
            # What each of these is asking to be thought about now, beside how
            # important the work is once it has been chosen. The workspace
            # prices deliberation's bid on `urgency` and this projection stated
            # none, so every goal she has ever held entered attention at the
            # same neutral default: nothing about what she was trying to do
            # could change what she attended to. The number is the engine's own
            # comparison — what it costs to leave each one alone — read per
            # goal rather than only for the one it picks.
            leaving = what_leaving_each_costs(shown)
            cognition.active_goals = [
                {
                    "id": item.get("id"),
                    "goal": item.get("objective") or item.get("name"),
                    "description": item.get("objective") or item.get("name"),
                    "status": item.get("status"),
                    "horizon": item.get("horizon"),
                    "priority": item.get("priority"),
                    "urgency": round(cost, 4),
                    "steps_done": item.get("steps_done"),
                    "steps_total": item.get("steps_total"),
                    "plan_id": item.get("plan_id"),
                    "task_id": item.get("task_id"),
                    "source": item.get("source"),
                }
                for item, cost in zip(shown, leaving, strict=True)
            ]
            current_objective = str(getattr(cognition, "current_objective", "") or "")
            current_origin = str(getattr(cognition, "current_origin", "") or "")
            actionable_labels = {
                str(item.get("objective") or item.get("name") or "")
                for item in active
            }
            if active and (
                not current_objective
                or is_intrinsic_goal_text(current_objective)
                or (
                    not _origin_is_user_anchored(current_origin)
                    and current_objective not in actionable_labels
                )
            ):
                cognition.current_objective = str(
                    _the_one_to_get_on_with(active).get("objective")
                    or _the_one_to_get_on_with(active).get("name")
                    or ""
                )
            elif not active and is_intrinsic_goal_text(current_objective):
                cognition.current_objective = None
        except (OSError, ConnectionError, TimeoutError) as exc:
            _record_goal_degradation(
                exc,
                severity="warning",
                action="left cognition.active_goals unchanged after state sync failure",
                classification=FallbackClassification.SILENT_LOSS_OF_CAPABILITY,
                extra={"phase": "state_sync"},
            )
            logger.debug("GoalEngine state sync skipped: %s", exc)

