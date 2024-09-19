try:
    import RPi.GPIO as GPIO
except ImportError:
    import sys
    import fake_rpi
    sys.modules['RPi'] = fake_rpi.RPi
    sys.modules['RPi.GPIO'] = fake_rpi.RPi.GPIO
    import RPi.GPIO as GPIO

import time

class SoilMoistureSensor:
    def __init__(self, analog_pin, digital_pin):
        self.analog_pin = analog_pin
        self.digital_pin = digital_pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.digital_pin, GPIO.IN)

    def read_analog(self):
        # Simulação de leitura analógica do sensor de umidade do solo
        value = 0  # Valor lido do ADC, substitua com leitura real 
        return value

    def is_soil_wet(self):
        return GPIO.input(self.digital_pin) == GPIO.LOW

    def check_soil_moisture(self):
        analog_value = self.read_analog()
        digital_status = self.is_soil_wet()

        print(f"Analog value: {analog_value}")

        if digital_status:
            print("Solo está úmido.")
        else:
            print("Solo está seco.")

        return analog_value, digital_status

    def cleanup(self):
        GPIO.cleanup()
