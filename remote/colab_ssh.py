# Colab cell — start an SSH endpoint on the GPU box.
#
# Paste this into a Colab notebook, run it, then send me the printed host:port.
# It installs sshd, sets a password, and opens a reverse tunnel, because Colab
# has no inbound route and the session is disposable.
#
# BEFORE YOU RUN IT: Runtime -> Change runtime type -> GPU.
#   T4 (free) works and is a real NVIDIA card, but it is sm_75 with NO bf16
#   tensor cores, so fla's KDA takes its non-MMA fallback. L4 or A100 exercise
#   the real path. Everything else in the comparison runs on T4 fine.
#
# The box can be killed whenever you like. Nothing is needed from it afterwards:
# the run script writes everything to a timestamped directory and pulls the
# logs back before it finishes, and re-running from scratch is cheap.

!apt-get update -qq && apt-get install -y -qq openssh-server > /dev/null 2>&1
!mkdir -p /run/sshd

# password for the tunnel, printed below
!echo 'root:probe' | chpasswd
!sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
!sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
!pkill sshd 2>/dev/null; !/usr/sbin/sshd -D -e &

# the tunnel. cloudflared is a single static binary, no account needed.
!curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared
!pkill cloudflared 2>/dev/null
!nohup cloudflared tunnel --url ssh://localhost:22 --no-autoupdate > /tmp/cf.log 2>&1 &

import time, re, subprocess
# wait for the tunnel to print its hostname
host = None
for _ in range(40):
    time.sleep(1)
    try:
        log = open("/tmp/cf.log").read()
    except Exception:
        continue
    m = re.search(r"trycloudflare\.com", log)
    if m:
        host = m.group(0)
        break

print()
print("=" * 70)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
print("=" * 70)
if host:
    print(f"SSH INTO IT WITH:")
    print(f"    ssh -p 22 root@{host}")
    print(f"password: probe")
    print("=" * 70)
else:
    print("tunnel did not come up; check /tmp/cf.log")
    print(subprocess.run(["tail", "-20", "/tmp/cf.log"], capture_output=True, text=True).stdout)
