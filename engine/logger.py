import time
import sys

class Timer:
 def __init__(self):
  self.start = time.time()
  
 def __call__(self) -> str:
  return f"[{time.time() - self.start:.0f}s]"

class TeeLogger:
 """
 Duplicates sys.stdout so that it writes to both the console and a specified log file.
 """
 def __init__(self, log_path: str):
  self.terminal = sys.stdout
  self.log = open(log_path, "a")
  
 def write(self, message: str):
  self.terminal.write(message)
  self.log.write(message)
  self.terminal.flush()
  self.log.flush()
  
 def flush(self):
  self.terminal.flush()
  self.log.flush()
