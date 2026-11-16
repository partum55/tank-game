#!/usr/bin/env python3
"""
Runs ONE agent file in its own process (started by the referee, not by hand).

Protocol, one JSON object per line over stdin/stdout:
    <- {"reset": 1}            -> {"ok": 1}
    <- {"obs": {...}}          -> {"action": [4 floats]}
    startup                    -> {"ready": 1}  or  {"error": "..."}
Anything the agent prints goes to stderr, so it cannot corrupt the protocol.
"""
import json
import os
import sys
import traceback


def main():
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    sys.stdout = sys.stderr

    def send(obj):
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    try:
        from tank_env.agent import load_agent, reset_agent, sanitize_action

        agent = load_agent(sys.argv[1])
    except BaseException:
        send({"error": traceback.format_exc()})
        return
    send({"ready": 1})

    for line in sys.stdin:
        try:
            msg = json.loads(line)
            if "reset" in msg:
                reset_agent(agent)
                send({"ok": 1})
            else:
                send({"action": sanitize_action(agent.act(msg["obs"]))})
        except BaseException:
            send({"error": traceback.format_exc()})
            return


if __name__ == "__main__":
    main()
