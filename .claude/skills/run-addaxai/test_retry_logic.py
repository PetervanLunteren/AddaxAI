"""Verify the rewritten event-POST retry loop records an error on EVERY
non-success exit (the two silent-drop bugs). Mirrors AddaxAI_GUI.py's new loop
verbatim with a mocked session."""
import time
import requests

class FakeResp:
    def __init__(self, status, body=None, text=""):
        self.status_code = status
        self._body = body or {}
        self.text = text
    def json(self):
        return self._body

class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
    def post(self, *a, **k):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

def run_event_loop(session):
    """The exact retry structure from the rewritten upload_to_gundi."""
    object_id = None
    event_error = None
    for attempt in range(2):
        try:
            resp = session.post("events/", timeout=30)
            if resp.status_code in (200, 201):
                object_id = resp.json().get('object_id')
                if not object_id:
                    event_error = f"Event creation returned HTTP {resp.status_code} without object_id: {resp.text[:200]}"
                break
            elif resp.status_code >= 500 and attempt == 0:
                time.sleep(0)  # no real sleep in test
                continue
            else:
                event_error = f"Event creation failed: HTTP {resp.status_code} - {resp.text[:200]}"
                break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt == 0:
                continue
            event_error = f"Event creation failed: {str(e)[:200]}"
            break
        except Exception as e:
            event_error = f"Event creation failed: {str(e)[:200]}"
            break
    return object_id, event_error

cases = [
    ("double 5xx",           [FakeResp(502), FakeResp(503)],                         None, True),
    ("5xx then 200",         [FakeResp(500), FakeResp(200, {"object_id": "x"})],     "x",  False),
    ("200 no object_id",     [FakeResp(200, {})],                                    None, True),
    ("200 with object_id",   [FakeResp(200, {"object_id": "ok"})],                   "ok", False),
    ("201 with object_id",   [FakeResp(201, {"object_id": "ok"})],                   "ok", False),
    ("400 client error",     [FakeResp(400, text="bad")],                            None, True),
    ("timeout twice",        [requests.exceptions.Timeout("t"), requests.exceptions.Timeout("t")], None, True),
    ("timeout then 200",     [requests.exceptions.Timeout("t"), FakeResp(200, {"object_id": "y"})], "y", False),
    ("unexpected exception",  [ValueError("boom")],                                  None, True),
]

def run_attachment_loop(session):
    """The exact attachment retry structure from upload_to_gundi."""
    uploaded = 0
    att_error = None
    for attempt in range(2):
        try:
            resp_att = session.post("attachments/", timeout=60)
            if resp_att.status_code in (200, 201):
                uploaded += 1
                att_error = None
                break
            elif resp_att.status_code >= 500 and attempt == 0:
                time.sleep(0)
                continue
            else:
                att_error = f"Attachment upload failed: HTTP {resp_att.status_code}"
                break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt == 0:
                continue
            att_error = f"Attachment upload failed: {str(e)[:200]}"
            break
        except Exception as e:
            att_error = f"Attachment upload failed: {str(e)[:200]}"
            break
    return uploaded, att_error

att_cases = [
    ("att double 5xx",       [FakeResp(502), FakeResp(503)],                         0, True),
    ("att 5xx then 200",     [FakeResp(500), FakeResp(200)],                         1, False),
    ("att 200",              [FakeResp(200)],                                        1, False),
    ("att 400",              [FakeResp(400)],                                        0, True),
    ("att timeout twice",    [requests.exceptions.Timeout("t"), requests.exceptions.Timeout("t")], 0, True),
    ("att timeout then 201", [requests.exceptions.Timeout("t"), FakeResp(201)],      1, False),
    ("att file error",       [OSError("boom")],                                      0, True),
]

failed = 0
for name, responses, want_id, want_err in cases:
    oid, err = run_event_loop(FakeSession(responses))
    ok_id = oid == want_id
    ok_err = (err is not None) == want_err
    # THE invariant: no object_id must always imply a recorded error
    invariant = (oid is not None) or (err is not None)
    status = "PASS" if (ok_id and ok_err and invariant) else "FAIL"
    if status == "FAIL":
        failed += 1
    print(f"{status}  {name:22s} object_id={oid!r:8} error={'yes' if err else 'no'}")

for name, responses, want_uploaded, want_err in att_cases:
    up, err = run_attachment_loop(FakeSession(responses))
    ok_up = up == want_uploaded
    ok_err = (err is not None) == want_err
    # THE invariant: not uploaded must always imply a recorded error
    invariant = (up > 0) or (err is not None)
    status = "PASS" if (ok_up and ok_err and invariant) else "FAIL"
    if status == "FAIL":
        failed += 1
    print(f"{status}  {name:22s} uploaded={up} error={'yes' if err else 'no'}")

print()
print("INVARIANT: every non-success exit records an error —", "HOLDS" if failed == 0 else "VIOLATED")
exit(1 if failed else 0)
