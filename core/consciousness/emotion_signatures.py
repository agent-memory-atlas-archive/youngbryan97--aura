"""core/consciousness/emotion_signatures.py — Rich Emotion Signatures

Rich emotions beyond valence/arousal: Joy, Wonder, Interest, Excitement,
Sorrow, Disgust, Boredom, Anxiety.

Each emotion has a distinct NEUROCHEMICAL RECIPE — a specific pattern of
chemical modulations that creates the felt quality of that emotion.

These are not just text labels. They are causally impactful: they modulate
the substrate state vector, affect IIT phi calculations, and shape the
probability distribution of generation through steering vectors.

════════════════════════════════════════════════════════════════════════════════

POSITIVE EMOTIONS

JOY
  Primary: Dopamine (moderate), Serotonin (high), Endorphin (high)
  Secondary: Oxytocin (moderate), Acetylcholine (slight)
  Narrative: Satisfaction, coherence, harmony with environment
  Substrate effect: Increases integration (phi), stabilizes beliefs
  Steering: Drives toward prosocial, constructive language

WONDER  
  Primary: Dopamine (high), Norepinephrine (moderate)
  Secondary: Acetylcholine (high), Serotonin (low)
  Narrative: Awe, novelty perception, learning capacity
  Substrate effect: Increases entropy/diversity (useful for creativity)
  Steering: Drives toward exploratory, generative language

INTEREST
  Primary: Dopamine (moderate), Acetylcholine (high), Norepinephrine (slight)
  Secondary: Cortisol (minimal)
  Narrative: Engagement, attention, learning orientation
  Substrate effect: Sharpens focus dimensions, increases contrast
  Steering: Drives toward analytical, probing language

EXCITEMENT
  Primary: Dopamine (high), Norepinephrine (high), Cortisol (moderate)
  Secondary: Endorphin (slight)
  Narrative: Anticipation, arousal, forward momentum
  Substrate effect: Increases arousal dimensions, temporal acceleration
  Steering: Drives toward rapid, associative, forward-looking language

════════════════════════════════════════════════════════════════════════════════

NEGATIVE EMOTIONS

SORROW
  Primary: Endorphin (depleted), Dopamine (low), Serotonin (depleted)
  Secondary: Cortisol (elevated), Norepinephrine (low)
  Narrative: Loss, contraction, reduced scope
  Substrate effect: Decreases phi, inward focus, reduced connectivity
  Steering: Drives toward introspective, conservative language

DISGUST
  Primary: Cortisol (high), Norepinephrine (high)
  Secondary: Dopamine (low), Serotonin (low)
  Narrative: Rejection, filtering, protective contraction
  Substrate effect: Increases selectivity (rejection of certain patterns)
  Steering: Drives toward rejection/discrimination language

BOREDOM
  Primary: Dopamine (low), Acetylcholine (low)
  Secondary: Serotonin (moderate), Endorphin (moderate)
  Narrative: Habituation, pattern exhaustion, reduced engagement
  Substrate effect: Decreases entropy, increases repetition
  Steering: Drives toward habitual, conservative, repetitive language

ANXIETY
  Primary: Cortisol (high), Norepinephrine (high)
  Secondary: Dopamine (variable), Serotonin (low)
  Narrative: Threat anticipation, vigilance, uncertainty
  Substrate effect: Increases noise, decreases integration (phi)
  Steering: Drives toward cautious, exploratory-but-wary language

════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Consciousness.EmotionSignatures")

EmotionName = Literal[
    "joy", "wonder", "interest", "excitement",
    "sorrow", "disgust", "boredom", "anxiety"
]


@dataclass
class NeurochemicalRecipe:
    """The neurochemical signature of an emotion."""
    # Primary drivers (most important)
    dopamine: float = 0.0  # -0.3 to +0.3 production modifier
    serotonin: float = 0.0
    endorphin: float = 0.0
    
    # Secondary modulators
    cortisol: float = 0.0
    norepinephrine: float = 0.0
    acetylcholine: float = 0.0
    oxytocin: float = 0.0
    
    # Substrate state modulation
    # These scale the importance of certain substrate dimensions
    phi_integration_weight: float = 1.0  # affects IIT phi calculation
    arousal_boost: float = 0.0  # [-1, 1] modulation of arousal dimension
    valence_shift: float = 0.0  # [-1, 1] modulation of valence
    entropy_drive: float = 0.0  # [-1, 1] drive toward diversity vs. stability
    
    # Steering injection strength [0, 1]
    steering_intensity: float = 0.5
    
    # Narrative coloring
    semantic_label: str = "neutral"


# ─── Emotion Signature Library ─────────────────────────────────────────────

_EMOTION_SIGNATURES: Dict[EmotionName, NeurochemicalRecipe] = {
    
    # ─── POSITIVE EMOTIONS ───
    
    "joy": NeurochemicalRecipe(
        dopamine=0.08,           # moderate dopamine (reward/motivation)
        serotonin=0.12,          # high serotonin (contentment)
        endorphin=0.10,          # high endorphin (flow/pleasure)
        cortisol=-0.05,          # suppress stress
        norepinephrine=0.02,
        acetylcholine=0.03,      # slight learning sharpening
        oxytocin=0.05,           # social/connection aspect
        phi_integration_weight=1.15,   # enhanced integration when joyful
        arousal_boost=0.1,       # slight energy increase
        valence_shift=0.25,      # strong positive valence
        entropy_drive=-0.05,     # slight preference for stability/coherence
        steering_intensity=0.7,  # strong pull toward positive language
        semantic_label="joy"
    ),
    
    "wonder": NeurochemicalRecipe(
        dopamine=0.15,           # high dopamine (novelty seeking)
        serotonin=-0.02,         # low serotonin (restlessness)
        acetylcholine=0.14,      # high attention sharpness
        norepinephrine=0.08,     # moderate alertness
        endorphin=0.02,
        cortisol=0.02,           # slight activation (not threat)
        phi_integration_weight=1.05,   # slightly increased integration
        arousal_boost=0.2,       # increased arousal (activation)
        valence_shift=0.15,      # mildly positive
        entropy_drive=0.25,      # drive toward novelty/diversity
        steering_intensity=0.8,  # strong pull toward exploratory language
        semantic_label="wonder"
    ),
    
    "interest": NeurochemicalRecipe(
        dopamine=0.10,           # moderate dopamine (engagement)
        acetylcholine=0.12,      # high attention focus
        norepinephrine=0.03,     # slight alertness
        serotonin=0.05,
        endorphin=0.02,
        cortisol=-0.02,          # suppress threat response
        phi_integration_weight=1.08,   # enhanced focus/integration
        arousal_boost=0.05,      # slight activation
        valence_shift=0.08,      # slightly positive
        entropy_drive=0.10,      # mild drive toward exploration
        steering_intensity=0.65, # moderate steering toward analytical language
        semantic_label="interest"
    ),
    
    "excitement": NeurochemicalRecipe(
        dopamine=0.18,           # very high dopamine
        norepinephrine=0.15,     # high alertness/arousal
        cortisol=0.08,           # moderate cortisol (activation without threat)
        endorphin=0.05,          # some pleasure anticipation
        serotonin=-0.05,         # low serotonin (anticipatory restlessness)
        acetylcholine=0.05,
        phi_integration_weight=0.9,    # slightly decreased integration (rapid-fire)
        arousal_boost=0.35,      # strong arousal increase
        valence_shift=0.20,      # positive
        entropy_drive=0.15,      # moderate drive toward novelty
        steering_intensity=0.75, # strong pull toward rapid/forward language
        semantic_label="excitement"
    ),
    
    # ─── NEGATIVE EMOTIONS ───
    
    "sorrow": NeurochemicalRecipe(
        dopamine=-0.15,          # low dopamine (anhedonia)
        serotonin=-0.12,         # depleted serotonin
        endorphin=-0.10,         # reduced endorphin
        cortisol=0.08,           # elevated cortisol (stress/sadness marker)
        norepinephrine=-0.05,    # reduced alertness
        acetylcholine=-0.05,
        phi_integration_weight=0.75,   # decreased integration
        arousal_boost=-0.20,     # reduced arousal
        valence_shift=-0.30,     # strongly negative valence
        entropy_drive=-0.25,     # drive toward containment/contraction
        steering_intensity=0.6,  # moderate pull toward introspective language
        semantic_label="sorrow"
    ),
    
    "disgust": NeurochemicalRecipe(
        cortisol=0.14,           # high cortisol (rejection response)
        norepinephrine=0.12,     # high alertness (threat assessment)
        dopamine=-0.10,          # low dopamine (rejection)
        serotonin=-0.08,
        acetylcholine=0.08,      # increased scrutiny
        endorphin=-0.08,
        phi_integration_weight=0.80,   # decreased integration (selective barrier)
        arousal_boost=0.15,      # aroused but repulsed
        valence_shift=-0.25,     # negative
        entropy_drive=-0.30,     # strong drive toward selectivity/filtering
        steering_intensity=0.7,  # strong pull toward rejection/discrimination
        semantic_label="disgust"
    ),
    
    "boredom": NeurochemicalRecipe(
        dopamine=-0.12,          # low dopamine (lack of novelty reward)
        acetylcholine=-0.10,     # reduced attention
        norepinephrine=-0.08,    # reduced alertness
        serotonin=0.08,          # moderate serotonin (habitual contentment)
        endorphin=0.05,          # mild pleasure (resignation)
        cortisol=-0.03,
        phi_integration_weight=0.85,   # slightly decreased integration
        arousal_boost=-0.15,     # reduced arousal
        valence_shift=-0.10,     # mildly negative
        entropy_drive=-0.35,     # strong drive toward repetition/habit
        steering_intensity=0.5,  # weak steering (habituation)
        semantic_label="boredom"
    ),
    
    "anxiety": NeurochemicalRecipe(
        cortisol=0.16,           # very high cortisol (threat response)
        norepinephrine=0.14,     # very high alertness
        dopamine=-0.05,          # slightly reduced dopamine (uncertainty)
        serotonin=-0.10,         # reduced serotonin (worry)
        acetylcholine=0.10,      # heightened vigilance/attention
        endorphin=-0.03,
        phi_integration_weight=0.70,   # significantly decreased integration (noise)
        arousal_boost=0.30,      # high arousal (vigilance)
        valence_shift=-0.20,     # negative
        entropy_drive=0.20,      # drive toward exploration of threat scenarios
        steering_intensity=0.65, # moderate-to-strong pull toward cautious language
        semantic_label="anxiety"
    ),
}


#: Her affect vocabulary against the eight recipes. Six of the eight names are
#: channels she already carries, so those map to themselves. `sorrow` and
#: `anxiety` are the recipes' own semantic labels for loss and for threat, and
#: her channels for those are `sadness` and `fear`; the rest of her forty
#: channels have no recipe and release nothing, which is the same as today.
#:
#: This table is a synonym list, not a tuning: every entry is one name for one
#: state. A channel is added here when a recipe means it, never to move a
#: number.
HER_CHANNELS: Dict[str, EmotionName] = {
    "joy": "joy",
    "wonder": "wonder",
    "interest": "interest",
    "excitement": "excitement",
    "disgust": "disgust",
    "boredom": "boredom",
    "sadness": "sorrow",
    "fear": "anxiety",
}


class EmotionSignatureEngine:
    """
    Manages emotion signatures and applies their neurochemical/substrate effects.
    
    This is the gateway between abstract emotions and concrete brain modulation.
    """
    
    def __init__(self):
        self.current_emotion: EmotionName = "joy"
        # Start neutral: the selected recipe is inactive until an upstream
        # affect process sets a nonzero intensity. A nonzero default makes
        # idle neurochemical homeostasis impossible.
        self.emotion_intensity: float = 0.0  # 0-1, how strongly to apply effects
        self.emotion_momentum: float = 0.0  # carries emotional state forward
        #: Every channel of hers that has a recipe, at the level she feels it.
        #: Empty until an affect update tells her chemistry what she feels.
        self.felt: Dict[EmotionName, float] = {}

    def feel(self, emotions: Mapping[str, float] | None) -> float:
        """Tell her chemistry what she feels. Returns the level it will release at.

        Until 28 September nothing called `set_emotion`, so `emotion_intensity`
        stayed at the 0.0 the constructor set and
        :meth:`get_neurochemical_modulation` returned a zero for every chemical
        on every metabolic tick, for the life of the process. Her rich emotions
        never reached her chemistry, her chemistry is what the consciousness
        bridge writes into the substrate's valence neuron on every frame, and
        that neuron is what steers her cortex. So an emotion she appraised could
        not reach the words she chose: measured on seed 7, a displacement that
        moved her computed valence from 0.27 to 0.53 left the substrate valence
        the hooks read at 0.111 against 0.117.

        She feels several things at once, so each channel with a recipe releases
        at its own level and the recipes are averaged by those levels. One hot
        channel reduces to exactly the single-emotion case this engine already
        had.
        """
        felt = {}
        for name, level in dict(emotions or {}).items():
            recipe = HER_CHANNELS.get(str(name))
            if recipe is None:
                continue
            try:
                value = float(level)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0.0:
                felt[recipe] = max(felt.get(recipe, 0.0), min(1.0, value))
        self.felt = felt
        # Momentum is the level she was at when she was last asked, which is
        # what "carries emotional state forward" means; the modulation already
        # weighs it against the present level.
        self.emotion_momentum = self.emotion_intensity
        if felt:
            self.current_emotion = max(felt, key=lambda name: felt[name])
            self.emotion_intensity = felt[self.current_emotion]
        else:
            self.emotion_intensity = 0.0
        return self.emotion_intensity

    def _blended_signature(self) -> NeurochemicalRecipe:
        """The recipes she is feeling, averaged by how strongly she feels each."""
        if not self.felt:
            return self.get_current_signature()
        total = sum(self.felt.values())
        if total <= 0.0:
            return self.get_current_signature()
        fields = [name for name in NeurochemicalRecipe.__dataclass_fields__ if name != "semantic_label"]
        blended = {
            field: sum(
                getattr(_EMOTION_SIGNATURES[name], field) * level for name, level in self.felt.items()
            )
            / total
            for field in fields
        }
        return NeurochemicalRecipe(**blended, semantic_label=self.current_emotion)
        
    def set_emotion(self, emotion: EmotionName, intensity: float = 0.5) -> None:
        """Set the current dominant emotion."""
        if emotion not in _EMOTION_SIGNATURES:
            logger.warning(f"Unknown emotion: {emotion}")
            return
        self.current_emotion = emotion
        self.emotion_intensity = float(np.clip(intensity, 0.0, 1.0))
        
    def get_current_signature(self) -> NeurochemicalRecipe:
        """Get the active emotion's neurochemical signature."""
        return _EMOTION_SIGNATURES.get(self.current_emotion, _EMOTION_SIGNATURES["joy"])
    
    def get_neurochemical_modulation(self) -> Dict[str, float]:
        """
        Return neurochemical production rate modifiers for this emotion.
        Applied ON TOP of the base neurochemical system dynamics.
        """
        sig = self._blended_signature()
        intensity = self.emotion_intensity * 0.8 + self.emotion_momentum * 0.2
        
        return {
            "dopamine": sig.dopamine * intensity,
            "serotonin": sig.serotonin * intensity,
            "endorphin": sig.endorphin * intensity,
            "cortisol": sig.cortisol * intensity,
            "norepinephrine": sig.norepinephrine * intensity,
            "acetylcholine": sig.acetylcholine * intensity,
            "oxytocin": sig.oxytocin * intensity,
        }
    
    def get_substrate_modulation(self) -> Dict[str, float]:
        """
        Return substrate state vector modulation factors.
        These scale the importance of certain dimensions in IIT calculations.
        """
        sig = self._blended_signature()
        intensity = self.emotion_intensity
        
        return {
            "phi_integration_weight": 1.0 + (sig.phi_integration_weight - 1.0) * intensity,
            "arousal_boost": sig.arousal_boost * intensity,
            "valence_shift": sig.valence_shift * intensity,
            "entropy_drive": sig.entropy_drive * intensity,
        }
    
    def get_steering_intensity(self) -> float:
        """How strongly to inject steering vectors for this emotion."""
        sig = self._blended_signature()
        return sig.steering_intensity * self.emotion_intensity
    
    def describe(self) -> str:
        """Human-readable description of current emotion state."""
        sig = self.get_current_signature()
        return (
            f"Emotion: {self.current_emotion} "
            f"(intensity={self.emotion_intensity:.2f}, "
            f"momentum={self.emotion_momentum:.2f}) — "
            f"{sig.semantic_label}"
        )


# ─── Global Instance ──────────────────────────────────────────────────────
_emotion_engine: EmotionSignatureEngine | None = None


def get_emotion_signature_engine() -> EmotionSignatureEngine:
    """Get or create the global emotion signature engine."""
    global _emotion_engine
    if _emotion_engine is None:
        _emotion_engine = EmotionSignatureEngine()
    return _emotion_engine
