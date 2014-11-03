import os
import sys
import asyncio
import time

def tail(fileName):
  file = open(fileName, 'r')
  file.seek(0, 2) # end
  while True:
    line = file.readline()
    if not line:
      time.sleep(0.1) # alternatively yield from asyncio.sleep(0.1)
      continue
    yield line

def getLastModificationTime(fileName):
  return os.stat(fileName).st_mtime

def getNewestFile(dir):
   files = (f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir,f)))
   fdLogFiles = (os.path.join(dir,f) for f in files if f.startswith('y'))
   newestLogFile = max((getLastModificationTime(f), f) for f in fdLogFiles)[1]
   return newestLogFile

dir = sys.argv[1] if len(sys.argv)>=2 else "C:/Users/kschubert/AppData/Local/MModal/DesktopDictationClient/logs"
for line in tail(getNewestFile(dir)):
  print(line,end="")

