/*
 * valve_controller.ino
 *
 * Arduino sketch for the MEG stroke-intervention solenoid valve controller.
 * Targets ATmega328P (Arduino Uno) and ESP32.
 *
 * Responsibilities:
 *   1. Receive MEG sensor data over serial.
 *   2. Run INT8 neural network inference (meg_inference).
 *   3. Drive three PWM-controlled solenoid valves.
 *   4. Enforce safety constraints (rate limiting, max extension, watchdog).
 *   5. Emit JSON-formatted telemetry over serial for monitoring.
 *
 * Pin assignments:
 *   PWM valve channels : D3  (valve 0), D5  (valve 1), D6 (valve 2)
 *   Status LED         : D13 (built-in)
 *   Heartbeat LED      : D9
 *
 * Serial protocol:
 *   Input  : 600 raw INT8 bytes (6 channels * 100 samples) per frame,
 *            or single-character commands: 'E' = emergency stop,
 *            'R' = reset, 'S' = status request.
 *   Output : JSON lines with inference results and system state.
 *
 * Safety:
 *   - Maximum valve extension capped at 80%.
 *   - Slew rate limited to 10%/s (~2.5 PWM units per 100ms tick).
 *   - Hardware watchdog timer (2s timeout).
 *   - Graceful degradation: if no serial data for 500ms, valves ramp to 0.
 */

#include "model_weights.h"

/* Forward declarations for the inference engine (arduino_inference.cpp) */
extern void meg_inference(const int8_t *input, float *output);

/* Watchdog */
#ifdef __AVR__
  #include <avr/wdt.h>
#elif defined(ESP32)
  #include "esp_task_wdt.h"
  #define WDT_TIMEOUT_S  2
#endif

/* Pin definitions */
static const uint8_t VALVE_PIN_0       = 3;
static const uint8_t VALVE_PIN_1       = 5;
static const uint8_t VALVE_PIN_2       = 6;
static const uint8_t LED_STATUS        = 13;
static const uint8_t LED_HEARTBEAT     = 9;

#define NUM_VALVES  3

static const uint8_t VALVE_PINS[NUM_VALVES] = {
    VALVE_PIN_0, VALVE_PIN_1, VALVE_PIN_2
};

/* Safety parameters */
static const float MAX_VALVE_FRACTION  = 0.80f;
static const float MAX_RATE_PER_SEC    = 0.10f;
static const unsigned long TICK_MS     = 100;
static const uint8_t MAX_PWM_VALUE     = (uint8_t)(MAX_VALVE_FRACTION * 255.0f);
static const float SLEW_PER_TICK       = MAX_RATE_PER_SEC
                                         * ((float)TICK_MS / 1000.0f)
                                         * 255.0f;
static const unsigned long SERIAL_TIMEOUT_MS = 500;

/* System state */
static uint8_t valve_pwm[NUM_VALVES]    = {0, 0, 0};
static uint8_t valve_target[NUM_VALVES] = {0, 0, 0};
static float inference_out[NUM_OUTPUTS]  = {0.0f, 0.0f, 0.0f};

/* MEG input buffer (6 channels * 100 samples = 600 bytes) */
static int8_t meg_input[NUM_MEG_CHANNELS * NUM_TIMEPOINTS];

static unsigned long last_serial_rx_ms  = 0;
static unsigned long last_tick_ms       = 0;
static unsigned long loop_count         = 0;
static bool emergency_stop = false;
static bool heartbeat_state = false;

/* Helper */
static inline int16_t clamp_i16(int16_t val, int16_t lo, int16_t hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

/* Safety: apply rate limiting and max extension cap */
static uint8_t apply_safety(uint8_t current, uint8_t target) {
    if (target > MAX_PWM_VALUE) target = MAX_PWM_VALUE;

    int16_t delta = (int16_t)target - (int16_t)current;
    int16_t max_delta = (int16_t)(SLEW_PER_TICK + 0.5f);
    if (max_delta < 1) max_delta = 1;

    delta = clamp_i16(delta, -max_delta, max_delta);

    int16_t result = (int16_t)current + delta;
    return (uint8_t)clamp_i16(result, 0, (int16_t)MAX_PWM_VALUE);
}

/* JSON telemetry */
static void send_json_status(void) {
    Serial.print("{\"t\":");
    Serial.print(millis());
    Serial.print(",\"v\":[");
    for (uint8_t i = 0; i < NUM_VALVES; i++) {
        Serial.print(valve_pwm[i]);
        if (i < NUM_VALVES - 1) Serial.print(",");
    }
    Serial.print("],\"p\":[");
    for (uint8_t i = 0; i < NUM_OUTPUTS; i++) {
        Serial.print(inference_out[i], 4);
        if (i < NUM_OUTPUTS - 1) Serial.print(",");
    }
    Serial.print("],\"e\":");
    Serial.print(emergency_stop ? 1 : 0);
    Serial.println("}");
}

/* Command handler */
static bool handle_command(uint8_t byte) {
    switch (byte) {
        case 'E':
            emergency_stop = true;
            for (uint8_t i = 0; i < NUM_VALVES; i++) {
                valve_pwm[i] = 0;
                valve_target[i] = 0;
                analogWrite(VALVE_PINS[i], 0);
            }
            Serial.println("{\"cmd\":\"ESTOP\",\"ok\":true}");
            return true;
        case 'R':
            emergency_stop = false;
            Serial.println("{\"cmd\":\"RESET\",\"ok\":true}");
            return true;
        case 'S':
            send_json_status();
            return true;
        default:
            return false;
    }
}

/* Read MEG frame from serial */
static bool read_meg_frame(void) {
    const int16_t frame_size = NUM_MEG_CHANNELS * NUM_TIMEPOINTS;  /* 600 */

    if (Serial.available() < frame_size) {
        while (Serial.available() > 0) {
            uint8_t b = Serial.peek();
            if (b == 'E' || b == 'R' || b == 'S') {
                Serial.read();
                handle_command(b);
            } else {
                break;
            }
        }
        return false;
    }

    for (int16_t i = 0; i < frame_size; i++) {
        meg_input[i] = (int8_t)Serial.read();
    }
    last_serial_rx_ms = millis();
    return true;
}

/* Watchdog helpers */
static void watchdog_init(void) {
#ifdef __AVR__
    wdt_enable(WDTO_2S);
#elif defined(ESP32)
    esp_task_wdt_init(WDT_TIMEOUT_S, true);
    esp_task_wdt_add(NULL);
#endif
}

static void watchdog_reset(void) {
#ifdef __AVR__
    wdt_reset();
#elif defined(ESP32)
    esp_task_wdt_reset();
#endif
}

void setup() {
    Serial.begin(115200);
    while (!Serial) { ; }

    for (uint8_t i = 0; i < NUM_VALVES; i++) {
        pinMode(VALVE_PINS[i], OUTPUT);
        analogWrite(VALVE_PINS[i], 0);
    }

    pinMode(LED_STATUS, OUTPUT);
    pinMode(LED_HEARTBEAT, OUTPUT);

    watchdog_init();

    last_tick_ms = millis();
    last_serial_rx_ms = millis();

    Serial.println("{\"event\":\"boot\",\"model\":\"MEGStrokeNet\","
                   "\"channels\":6,\"timepoints\":100,\"outputs\":3}");

    for (uint8_t i = 0; i < 3; i++) {
        digitalWrite(LED_STATUS, HIGH);
        delay(100);
        digitalWrite(LED_STATUS, LOW);
        delay(100);
    }
}

void loop() {
    watchdog_reset();
    unsigned long now = millis();

    bool got_frame = read_meg_frame();

    if (got_frame && !emergency_stop) {
        meg_inference(meg_input, inference_out);

        for (uint8_t i = 0; i < NUM_VALVES; i++) {
            float p = inference_out[i];
            if (p < 0.0f) p = 0.0f;
            if (p > 1.0f) p = 1.0f;
            valve_target[i] = (uint8_t)(p * 255.0f);
        }
    }

    /* Graceful degradation on signal loss */
    if ((now - last_serial_rx_ms) > SERIAL_TIMEOUT_MS && !emergency_stop) {
        for (uint8_t i = 0; i < NUM_VALVES; i++) {
            valve_target[i] = 0;
        }
        if ((now / 250) % 2 == 0) {
            digitalWrite(LED_STATUS, HIGH);
        } else {
            digitalWrite(LED_STATUS, LOW);
        }
    }

    /* Fixed-rate control tick */
    if ((now - last_tick_ms) >= TICK_MS) {
        last_tick_ms = now;

        if (!emergency_stop) {
            for (uint8_t i = 0; i < NUM_VALVES; i++) {
                valve_pwm[i] = apply_safety(valve_pwm[i], valve_target[i]);
                analogWrite(VALVE_PINS[i], valve_pwm[i]);
            }
        }

        send_json_status();

        heartbeat_state = !heartbeat_state;
        digitalWrite(LED_HEARTBEAT, heartbeat_state ? HIGH : LOW);

        if (!emergency_stop && (now - last_serial_rx_ms) <= SERIAL_TIMEOUT_MS) {
            digitalWrite(LED_STATUS, HIGH);
        }

        loop_count++;
    }
}
