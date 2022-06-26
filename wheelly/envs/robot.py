import socket
import re
import time
import logging

from matplotlib.colors import NoNorm

logger = logging.getLogger(__name__)

CONNECTION_TIMEOUT = 10
READ_TIMEOUT = 1

class Robot:
    def __init__(self, params):
        """Create a Robot object to comunicate to robot
        
        Arguments:

        params -- dict the dictionary with the paramters
        """
        self._host = params["host"]
        self._port = params["port"]
        self._connection_timeout = params["connectionTimeout"] if "connectionTimeout" in params else CONNECTION_TIMEOUT
        self._read_timeout = params["readTimeout"] if "readTimeout" in params else READ_TIMEOUT
        self._socket = None
        self._timestamp_offset = None

    def connect(self):
        """Connect the robot socket"""
        if self._socket == None:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self._connection_timeout)
            self._socket.connect((self._host, self._port))
            self._file = self._socket.makefile("rb")

    def sync(self):
        now = time.time()
        ref = f"{now}"
        self.write_cmd(f"ck {ref}")
        while True:
            clock = self.parse_clock(self.read_line())
            if clock and clock["ref"] == ref:
                break
        rp = float(clock["reply"] - clock["received"]) / 1000;
        elaps = clock["timestamp"] - now
        latency = (elaps - rp) / 2
        # offset + received - latency = now 
        # offset = now - received + latency
        offset = now - float(clock["received"]) / 1000 + latency;
        self._timestamp_offset = offset

    def write_cmd(self, cmd):
        """Write a command string to robot socket and return the command string.
        
        Arguments:
        cmd -- the command string
        """
        if self._socket:
            self._socket.settimeout(None)
            line = cmd + "\n"
            logger.debug("--> %s", cmd)
            self._socket.sendall(line.encode("utf-8"))
            return cmd

    def move(self, dir, speed):
        """Write a move command to robot socket and return the command string.
        
        Arguments:
        dir -- int the direction (DEG)
        speed -- float the speed (-1 ... 1)
        """
        return self.write_cmd(f"mv {dir} {speed}")

    def scan(self, dir):
        """Write a scan command to robot socket and return the command string.
        
        Arguments:
        dir -- int the sensor direction (DEG)
        """
        return self.write_cmd(f"sc {dir}")

    def halt(self):
        """Write a halt command to robot socket and return the command string."""
        return self.write_cmd("al")

    def read_line(self):
        """Read a line from robot socket and return a string"""
        if self._file:
            self._socket.settimeout(self._read_timeout)
            data  = self._file.readline()
            timestamp = time.time()
            line = data.decode("utf-8")
            logger.debug("<-- %s", line[0: -1])
            return (line, timestamp)
        else:
            return None

    def read_status(self):
        """Read the status from robot socket and return the status"""
        if self._socket:
            if self._timestamp_offset == None:
                self.sync()
            data = self.read_line()
            return self.parse_status(data) if data else None
        else:
            return None

    def close(self):
        """Close robot socket"""
        if self._socket:
            self._socket.close()

    # ck 1656250845.8646197 12387 12389
    # ck ref received_instant reply_instant
    def parse_clock(self, timed_line):
        """Parse the clock line and return the clock dictionary
    
        Argument:
        line -- (string) the clock line
        """
        m = re.search(r"ck (.*) (\d*) (\d*)", timed_line[0])
        return {
            "timestamp": timed_line[1],
            "ref": m.group(1),
            "received": int(m.group(2)),
            "reply": int(m.group(3)),
        } if m else None

    # st 36517 0.000 0.000 1 0 0.22 0.000 0.000 0 13.12 1 1 0 1 0 0.00 0
    # st clk x y deg sens dist left right contacs voltage canMoveForw camMoveBack imuFailure halt moveDir moveSpeed nextSensor
    def parse_status(self, timed_line):
        """Parse the status line and return the status dictionary
    
        Arguments:
        line -- string the status line
        timestamp -- float the timestamp of message
        """
        m = re.search(r"st (\d*) (-?\d*\.\d*) (-?\d*\.\d*) (-?\d*) (-?\d*) (-?\d*\.\d*) (-?\d*\.\d*) (-?\d*\.\d*) (\d*) (-?\d*\.\d*) ([01]) ([01]) ([01]) ([01]) (-?\d*) (-?\d*\.\d*) (-?\d*)", timed_line[0])
        return {
            "timestamp": float(m.group(1)) / 1000 + self._timestamp_offset,
            "x": float(m.group(2)),
            "y": float(m.group(3)),
            "dir": int(m.group(4)),
            "sensor": int(m.group(5)),
            "dist": float(m.group(6)),
            "left": float(m.group(7)),
            "right": float(m.group(8)),
            "contacts": int(m.group(9)),
            "canMoveForward": int(m.group(11)),
            "canMoveBackward": int(m.group(12)),
        } if m else None
