"""
simulation_main.py — Run the visual simulation (no camera, no models needed).
Demonstrates the core threat detection logic with animated dogs and humans.

This is the FIRST thing to run — proves the logic works before adding hardware.

Controls:
  1-6   : Switch between test scenarios
  SPACE : Pause / Resume
  R     : Reset current scenario
  Q     : Quit

Scenarios:
  1. Aggressive dog ALONE         → NO ultrasonic (proves Rule 2)
  2. Aggressive dog → Human       → ALERT → DANGER → EMIT (proves Rule 3)
  3. Calm dog near human          → IDLE always (no false positives)
  4. Dog fight (2 dogs)           → DOG_FIGHT → NEVER trigger (proves Rule 1)
  5. Multiple dogs + human        → Only aggressive → human triggers
  6. Full attack sequence         → Dog chases human, human flees
"""

import time
import cv2

from config import SIM_FPS, WINDOW_NAME
from simulation.simulator import Simulator
from simulation.scenarios import ScenarioManager, update_scenario_behavior
from utils.logger import EventLogger


def run_simulation():
    print("=" * 60)
    print("  SIMULATION MODE — Smart Dog Threat Detection")
    print("=" * 60)
    print()
    print("  No camera or models needed!")
    print("  Testing threat logic with animated scenarios.")
    print()
    print("  Controls:")
    print("    1-6   : Switch scenario")
    print("    SPACE  : Pause / Resume")
    print("    R      : Reset scenario")
    print("    Q      : Quit")
    print()

    # Print all scenarios
    for sid, name in ScenarioManager.SCENARIOS.items():
        print(f"    [{sid}] {name}")
    print()
    print("=" * 60)

    sim = Simulator()
    logger = EventLogger()

    # Start with scenario 1
    current_scenario = 1
    scenario_name = ScenarioManager.load(sim, current_scenario)
    print(f"\n[SIM] Loaded: {scenario_name}")

    frame_delay = 1.0 / SIM_FPS

    try:
        while True:
            start_time = time.time()

            # Update entity positions
            if not sim.paused:
                update_scenario_behavior(sim, current_scenario)
                sim.frame_count += 1

            # Classify threats through the REAL engine
            result, cnn_results, audio_state = sim.classify_threats()

            # Update ultrasonic state
            dt = frame_delay if not sim.paused else 0
            sim.update_ultrasonic(result, dt)

            # Render
            canvas = sim.render()
            canvas = sim.draw_hud(canvas, result, audio_state, scenario_name)

            # Log
            logger.log(
                num_dogs=len(sim.get_dogs()),
                num_humans=len(sim.get_humans()),
                threat_class=result["threat_class"],
                threat_label=result["threat_label"],
                confidence=result["confidence"],
                audio_bark=audio_state.get("bark", False),
                audio_growl=audio_state.get("growl", False),
                audio_scream=audio_state.get("scream", False),
                ultrasonic_triggered=result["trigger_ultrasonic"],
                notes=f"[SIM-{current_scenario}] {result['reason']}",
            )

            # Display
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF

            # Controls
            if key == ord("q"):
                print("\n[SIM] Quit requested")
                break
            elif key == ord(" "):
                sim.paused = not sim.paused
                print(f"[SIM] {'Paused' if sim.paused else 'Resumed'}")
            elif key == ord("r"):
                scenario_name = ScenarioManager.load(sim, current_scenario)
                print(f"[SIM] Reset: {scenario_name}")
            elif ord("1") <= key <= ord("6"):
                current_scenario = key - ord("0")
                scenario_name = ScenarioManager.load(sim, current_scenario)
                print(f"\n[SIM] Switched to Scenario {current_scenario}: {scenario_name}")

            # Frame rate control
            elapsed = time.time() - start_time
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[SIM] Interrupted")

    finally:
        cv2.destroyAllWindows()
        logger.close()

        # Print event log summary
        if sim.event_log:
            print("\n  Event Log:")
            for event in sim.event_log:
                print(f"    {event}")

        print(f"\n  Total ultrasonic triggers: {sim.ultrasonic_total}")
        print("[SIM] Done!")


if __name__ == "__main__":
    run_simulation()
