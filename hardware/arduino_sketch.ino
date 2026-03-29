/*
 * arduino_sketch.ino — Arduino Ultrasonic Emitter Controller
 * Smart Dog Threat Detection System
 *
 * Receives serial commands from Python:
 *   "ULTRASONIC_ON"  → activates ultrasonic emitter on Pin 9
 *   "ULTRASONIC_OFF" → deactivates ultrasonic emitter
 *   "STATUS"         → replies with current state
 *
 * Wiring:
 *   Pin 9  → Ultrasonic transducer signal (+)
 *   GND    → Ultrasonic transducer ground (-)
 *   Pin 13 → Status LED (built-in, blinks when emitting)
 *
 * Ultrasonic frequency: ~25kHz via Timer1 PWM
 */

#define ULTRASONIC_PIN 9    // OC1A — Timer1 PWM output
#define LED_PIN 13          // Built-in LED for status
#define SERIAL_BAUD 9600

bool emitting = false;
String inputBuffer = "";

void setup() {
    Serial.begin(SERIAL_BAUD);
    pinMode(ULTRASONIC_PIN, OUTPUT);
    pinMode(LED_PIN, OUTPUT);

    digitalWrite(ULTRASONIC_PIN, LOW);
    digitalWrite(LED_PIN, LOW);

    Serial.println("READY");
}

void loop() {
    // Read serial commands
    while (Serial.available() > 0) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            inputBuffer.trim();
            if (inputBuffer.length() > 0) {
                processCommand(inputBuffer);
            }
            inputBuffer = "";
        } else {
            inputBuffer += c;
        }
    }

    // Blink LED when emitting
    if (emitting) {
        digitalWrite(LED_PIN, (millis() / 100) % 2);
    }
}

void processCommand(String cmd) {
    if (cmd == "ULTRASONIC_ON") {
        startUltrasonic();
        Serial.println("ACK:ON");
    }
    else if (cmd == "ULTRASONIC_OFF") {
        stopUltrasonic();
        Serial.println("ACK:OFF");
    }
    else if (cmd == "STATUS") {
        Serial.println(emitting ? "EMITTING" : "OK");
    }
    else {
        Serial.println("ERR:UNKNOWN_CMD");
    }
}

void startUltrasonic() {
    /*
     * Configure Timer1 for ~25kHz PWM on Pin 9:
     *   16MHz / (2 * 1 * 320) = 25kHz
     *   Using Phase-correct PWM, prescaler=1, TOP=320
     *   50% duty cycle: OCR1A = 160
     */
    TCCR1A = _BV(COM1A1) | _BV(WGM11);       // Phase-correct PWM
    TCCR1B = _BV(WGM13) | _BV(CS10);          // Prescaler = 1
    ICR1 = 320;                                 // TOP = 320 → 25kHz
    OCR1A = 160;                                // 50% duty cycle

    emitting = true;
    digitalWrite(LED_PIN, HIGH);
}

void stopUltrasonic() {
    // Disable Timer1 PWM
    TCCR1A = 0;
    TCCR1B = 0;
    ICR1 = 0;
    OCR1A = 0;

    digitalWrite(ULTRASONIC_PIN, LOW);
    emitting = false;
    digitalWrite(LED_PIN, LOW);
}
