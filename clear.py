import naoqi

# Create an instance of the ALBroker
broker = naoqi.ALBroker("broker", "0.0.0.0", 0, "192.168.0.130", 9559)

# Unregister the module/service
broker.shutdown()
