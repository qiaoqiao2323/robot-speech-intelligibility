import qi

# Connect to the NAO robot
ip_address = "192.168.0.120"  # Replace with the actual IP address of your NAO robot
port = 9559  # Default port for NAOqi
session = qi.Session()
session.connect("tcp://" + ip_address + ":" + str(port))

# Get the text-to-speech engine
tts = session.service("ALTextToSpeech")

# Get the current speech engine
current_engine = tts.getVoice()

# Print the current speech engine
print("Current Speech Engine:", current_engine)

# Disconnect from the NAO robot
session.close()
