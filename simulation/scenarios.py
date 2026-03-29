"""
simulation/scenarios.py — Pre-built test scenarios for the simulator.
Each scenario demonstrates a specific threat case.

Scenarios:
  1. Aggressive dog ALONE         → should NOT emit ultrasonic
  2. Aggressive dog approaches    → ALERT, then DANGER + EMIT
  3. Calm dog near human          → IDLE, no trigger
  4. Dog fight (2 dogs)           → DOG_FIGHT, NEVER trigger
  5. Multiple dogs + human        → only trigger if dog attacks human
  6. Full attack sequence         → dog spawns far, runs at human, triggers
"""

from config import SIM_WIDTH, SIM_HEIGHT, SIM_DOG_SPEED, SIM_HUMAN_SPEED


class ScenarioManager:
    """Manages and loads simulation scenarios."""

    SCENARIOS = {
        1: "Aggressive Dog ALONE (NO ultrasonic)",
        2: "Aggressive Dog → Human (EMIT ultrasonic)",
        3: "Calm Dog Near Human (no threat)",
        4: "Dog Fight — 2 Dogs (NEVER trigger)",
        5: "Multiple Dogs + Human (selective trigger)",
        6: "Full Attack Sequence (approach → attack)",
    }

    @staticmethod
    def load(sim, scenario_id):
        """Load a scenario into the simulator."""
        sim.clear()

        if scenario_id == 1:
            _scenario_aggressive_alone(sim)
        elif scenario_id == 2:
            _scenario_aggressive_approaches_human(sim)
        elif scenario_id == 3:
            _scenario_calm_dog_near_human(sim)
        elif scenario_id == 4:
            _scenario_dog_fight(sim)
        elif scenario_id == 5:
            _scenario_multiple_dogs_human(sim)
        elif scenario_id == 6:
            _scenario_full_attack(sim)

        return ScenarioManager.SCENARIOS.get(scenario_id, "Unknown")

    @staticmethod
    def get_name(scenario_id):
        return ScenarioManager.SCENARIOS.get(scenario_id, "Unknown")


def _scenario_aggressive_alone(sim):
    """
    Scenario 1: Aggressive dog alone on screen.
    Expected: IDLE (aggressive but no human), NEVER trigger ultrasonic.
    This tests Rule 2: No humans → NEVER classify DANGER.
    """
    dog = sim.add_dog(SIM_WIDTH // 2, SIM_HEIGHT // 2, aggressive=True, name="Angry Dog")
    dog.wander(SIM_DOG_SPEED)


def _scenario_aggressive_approaches_human(sim):
    """
    Scenario 2: Aggressive dog on left, human on right.
    Dog moves toward human. Should go IDLE → ALERT → DANGER → EMIT.
    """
    dog = sim.add_dog(100, SIM_HEIGHT // 2, aggressive=True, name="Attack Dog")
    human = sim.add_human(SIM_WIDTH - 150, SIM_HEIGHT // 2, name="Victim")
    dog.target = human


def _scenario_calm_dog_near_human(sim):
    """
    Scenario 3: Calm (non-aggressive) dog near a human.
    Expected: IDLE always, no trigger.
    """
    sim.add_dog(350, SIM_HEIGHT // 2, aggressive=False, name="Friendly Dog")
    sim.add_human(450, SIM_HEIGHT // 2, name="Dog Owner")


def _scenario_dog_fight(sim):
    """
    Scenario 4: Two aggressive dogs fighting each other.
    Expected: DOG_FIGHT, Rule 1: NEVER trigger ultrasonic.
    """
    dog1 = sim.add_dog(300, SIM_HEIGHT // 2, aggressive=True, name="Dog A")
    dog2 = sim.add_dog(500, SIM_HEIGHT // 2, aggressive=True, name="Dog B")
    dog1.target = dog2
    dog2.target = dog1

    # Add a human far away to show that even with human present,
    # DOG_FIGHT still doesn't trigger
    sim.add_human(SIM_WIDTH - 100, 100, name="Bystander")


def _scenario_multiple_dogs_human(sim):
    """
    Scenario 5: One aggressive dog, one calm dog, one human.
    Only the aggressive dog approaching human should trigger.
    """
    agg_dog = sim.add_dog(100, 200, aggressive=True, name="Aggressive")
    sim.add_dog(100, 400, aggressive=False, name="Calm Dog")
    human = sim.add_human(SIM_WIDTH - 150, SIM_HEIGHT // 2, name="Person")
    agg_dog.target = human


def _scenario_full_attack(sim):
    """
    Scenario 6: Full attack sequence — dog starts far away, runs toward human.
    Demonstrates the complete pipeline:
    - Far away: IDLE (even though aggressive)
    - Approaching: ALERT
    - Close range: DANGER → ULTRASONIC TRIGGERED
    - Human runs away, dog chases
    """
    dog = sim.add_dog(50, SIM_HEIGHT // 2, aggressive=True, name="Attacker")
    human = sim.add_human(SIM_WIDTH - 100, SIM_HEIGHT // 2, name="Runner")
    dog.target = human
    # Human will try to flee (handled in update loop)


def update_scenario_behavior(sim, scenario_id):
    """
    Per-frame behavior update for each scenario.
    Moves entities according to scenario logic.
    """
    dogs = sim.get_dogs()
    humans = sim.get_humans()

    if scenario_id == 1:
        # Aggressive dog wanders alone
        for dog in dogs:
            dog.wander(SIM_DOG_SPEED)
            dog.move()

    elif scenario_id == 2:
        # Aggressive dog moves toward human
        for dog in dogs:
            if dog.target:
                dog.move_toward(dog.target, SIM_DOG_SPEED)
            dog.move()

    elif scenario_id == 3:
        # Calm dog and human wander near each other
        for dog in dogs:
            dog.wander(SIM_DOG_SPEED * 0.3)
            dog.move()
        for human in humans:
            human.wander(SIM_HUMAN_SPEED * 0.3)
            human.move()

    elif scenario_id == 4:
        # Two dogs approach each other (fight)
        for dog in dogs:
            if dog.target:
                dog.move_toward(dog.target, SIM_DOG_SPEED * 0.8)
            dog.move()

    elif scenario_id == 5:
        # Aggressive dog approaches human, calm dog wanders
        for dog in dogs:
            if dog.aggressive and dog.target:
                dog.move_toward(dog.target, SIM_DOG_SPEED)
            elif not dog.aggressive:
                dog.wander(SIM_DOG_SPEED * 0.3)
            dog.move()

    elif scenario_id == 6:
        # Full attack: dog chases, human flees when dog is close
        for dog in dogs:
            if dog.target:
                dog.move_toward(dog.target, SIM_DOG_SPEED * 1.3)
            dog.move()

        for human in humans:
            # Human runs away from nearest aggressive dog
            nearest_agg = None
            min_dist = float("inf")
            for dog in dogs:
                if dog.aggressive:
                    d = dog.distance_to(human)
                    if d < min_dist:
                        min_dist = d
                        nearest_agg = dog
            if nearest_agg and min_dist < 200:
                human.move_away_from(nearest_agg, SIM_HUMAN_SPEED * 1.5)
            else:
                human.wander(SIM_HUMAN_SPEED * 0.3)
            human.move()
