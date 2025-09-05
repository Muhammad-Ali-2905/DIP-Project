import cv2
import socket
import pickle
import struct

# Connect to Raspberry Pi
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.24.246', 8485))  # Replace with Pi's IP

data = b""  # To store incoming data
payload_size = struct.calcsize("Q")  # Size of the packed message size (8 bytes)

while True:
    # Read message length (the first 8 bytes)
    while len(data) < payload_size:
        packet = client_socket.recv(4096)
        if not packet:
            break
        data += packet
    if len(data) < payload_size:
        break
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]

    try:
        msg_size = struct.unpack("Q", packed_msg_size)[0]
    except struct.error as e:
        print(f"Error unpacking message size: {e}")
        break

    # Read frame data
    while len(data) < msg_size:
        data += client_socket.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Deserialize the frame and display it
    frame = pickle.loads(frame_data)
    cv2.imshow("Video from Pi", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        print("ESC pressed. Closing connection.")
        break

client_socket.close()
cv2.destroyAllWindows()
