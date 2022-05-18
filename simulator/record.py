import GPUtil
import psutil
import time
import logging
import re
import sys
LOG_FORMAT = "%(asctime)s %(message)s"
DATE_FORMAT = "%Y/%m/%d %H:%M:%S"

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

targetNIC = "enp3s0"

pre_bits_recv = psutil.net_io_counters(pernic=True)[targetNIC].bytes_recv * 8
pre_bits_sent = psutil.net_io_counters(pernic=True)[targetNIC].bytes_sent * 8

time.sleep(1)

cur_bits_recv = psutil.net_io_counters(pernic=True)[targetNIC].bytes_recv * 8
cur_bits_sent = psutil.net_io_counters(pernic=True)[targetNIC].bytes_sent * 8

targetGPU = GPUtil.getGPUs()[0]

targetPID = []
r = re.compile(".*grpcserver")
s = re.compile("./run.sh")
doing = False
for process in psutil.process_iter():
    cmdline = process.cmdline()
    if len(list(filter(r.match, process.cmdline()))):
        targetPID.append(process.pid)
    if len(list(filter(s.match,process.cmdline()))):
        doing  = True
if not targetPID:
    exit
if not doing:
    #logging.debug("DONE")
    sys.exit(43)
mem = 0
for pid in targetPID:
    p = psutil.Process(pid)
    mem += p.memory_info().rss / psutil.virtual_memory().total

log = "[DEBUG] STATS. CPU: %.2f, MEM: %.2f, GPU: %.2f, GPUMEM: %.2f, UPLINK: %.2f, DOWNLINK: %.2f, SENT: %.2f, RECV: %.2f" % (
    psutil.cpu_percent()/100,
    mem,
    targetGPU.load,
    targetGPU.memoryUtil,
    (cur_bits_sent - pre_bits_sent) /(1024), # in Kbps
    (cur_bits_recv - pre_bits_recv) /(1024), # in Kbps
    (pre_bits_sent) / 1024, # in Kbits
    (pre_bits_recv) / 1024  # in Kbits
)

# print(log)
logging.debug(log)
