"""AddaxAI GUI driver — launches the real app, probes and drives the UI, exits.

Runs AddaxAI_GUI.py in-process with a patched Tk.mainloop that schedules probe
actions after startup. This gives programmatic access to the live widget tree
(tkinter has no external automation surface like Playwright).

Usage (from the AddaxAI repo root):
    .venv/bin/python .claude/skills/run-addaxai/driver.py            # probe + interact + quit
    KEEP_OPEN=1 .venv/bin/python .claude/skills/run-addaxai/driver.py  # leave window open for a human

Exit code 0 = app launched, widgets found, Gundi checkbox toggled the options
frame correctly. Non-zero = a probe failed (details on stdout).

Screenshots: tries macOS `screencapture` into this directory; that needs the
terminal to have Screen Recording permission (System Settings > Privacy &
Security > Screen Recording) and fails gracefully without it.
"""
import os
import subprocess
import sys
import runpy
import tkinter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
KEEP_OPEN = os.environ.get("KEEP_OPEN", "") not in ("", "0")

results = {"failures": []}

def fail(msg):
    print(f"FAIL {msg}")
    results["failures"].append(msg)

def ok(msg):
    print(f"OK   {msg}")

def walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from walk(child)

def find_label(root, text):
    for w in walk(root):
        try:
            if isinstance(w, tkinter.Label) and w.cget("text") == text:
                return w
        except Exception:
            pass
    return None

def sibling_checkbutton(label):
    """The Checkbutton gridded on the same row as a Label, in the same master."""
    row = label.grid_info().get("row")
    for w in label.master.winfo_children():
        if isinstance(w, tkinter.Checkbutton) and w.grid_info().get("row") == row:
            return w
    return None

def find_labelframe_containing(root, fragment):
    for w in walk(root):
        try:
            if isinstance(w, tkinter.LabelFrame) and fragment in str(w.cget("text")):
                return w
        except Exception:
            pass
    return None

def screenshot(root, name):
    """Capture ONLY the app window region (never the full desktop)."""
    path = os.path.join(HERE, name)
    try:
        x, y = root.winfo_rootx(), root.winfo_rooty()
        # customtkinter's CTk root misreports winfo_width/height (scaling
        # quirk: stays at the 200x200 default) — pad the region so the whole
        # window is captured; stays gitignored either way
        w = max(root.winfo_width(), 900)
        h = max(root.winfo_height(), 950)
        region = f"{x},{y},{w},{h}"
        r = subprocess.run(["screencapture", "-x", f"-R{region}", path],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and os.path.isfile(path):
            ok(f"screenshot saved ({region}): {path}")
        else:
            print(f"NOTE screenshot unavailable ({(r.stderr or 'no output').strip()}) — "
                  "grant Screen Recording permission to your terminal to enable")
    except Exception as e:
        print(f"NOTE screenshot unavailable ({e})")

def find_ctk_button(root, text):
    """Find a customtkinter CTkButton by its text."""
    for w in walk(root):
        try:
            if type(w).__name__ in ("CTkButton", "GreyTopButton") and w.cget("text") == text:
                return w
        except Exception:
            pass
    return None

def probe(root):
    try:
        ok(f"window up: title={root.wm_title()!r} geometry={root.winfo_geometry()}")

        # switch to advanced mode so the 4th-step pane (with the Gundi
        # controls) is actually displayed — the app starts in simple mode
        adv_btn = find_ctk_button(root, "To advanced mode")
        if adv_btn is not None:
            try:
                adv_btn.invoke()
            except Exception:
                adv_btn._clicked(None)  # older customtkinter has no invoke()
            root.update()
            ok("switched to advanced mode")
        else:
            print("NOTE 'To advanced mode' button not found (already advanced?)")

        # find the Gundi checkbox via its label
        lbl = find_label(root, "Upload events to Gundi")
        if lbl is None:
            fail("could not find 'Upload events to Gundi' label in widget tree")
            return
        ok("found Gundi upload label")
        chb = sibling_checkbutton(lbl)
        if chb is None:
            fail("could not find Gundi checkbox next to its label")
            return

        frame = find_labelframe_containing(root, "Gundi options")
        if frame is None:
            fail("could not find 'Gundi options' frame")
            return

        # diagnostics
        print(f"     chb state={chb.cget('state')!r} var={root.getvar(str(chb.cget('variable')))!r} "
              f"root wm_geometry={root.wm_geometry()!r} req={root.winfo_reqwidth()}x{root.winfo_reqheight()}")

        # step-4 widgets start disabled until a folder is chosen: a disabled
        # Checkbutton silently ignores invoke(), and toggle_gundi_frame guards
        # on the LABEL's state ('normal' == step enabled). enable both to
        # simulate an enabled step 4
        for w in (chb, lbl):
            if str(w.cget('state')) == 'disabled':
                w.configure(state='normal')

        # winfo_manager() reports whether toggle_gundi_frame gridded the frame
        # ('grid') or grid_forget it (''); works regardless of pane visibility
        before = frame.winfo_manager()
        chb.invoke()  # real click: toggles the var AND runs toggle_gundi_frame
        root.update()
        after = frame.winfo_manager()
        if before == after:
            fail(f"Gundi options frame did not change on toggle (manager stayed {before!r})")
        else:
            ok(f"Gundi checkbox toggles options frame (manager {before!r} -> {after!r})")

        screenshot(root, "screenshot-addaxai.png")

        chb.invoke()  # restore original state
        root.update()
    except Exception as e:
        fail(f"probe raised: {e!r}")
    finally:
        if not KEEP_OPEN:
            root.after(200, root.destroy)

# patch mainloop so the probe runs once the app is fully constructed
_orig_mainloop = tkinter.Tk.mainloop
def _probing_mainloop(self, *args, **kwargs):
    self.after(2500, probe, self)
    _orig_mainloop(self, *args, **kwargs)
tkinter.Tk.mainloop = _probing_mainloop

os.chdir(REPO)
sys.path.insert(0, REPO)
try:
    runpy.run_path(os.path.join(REPO, "AddaxAI_GUI.py"), run_name="__main__")
except SystemExit:
    pass

print()
if results["failures"]:
    print(f"DRIVER RESULT: {len(results['failures'])} failure(s)")
    sys.exit(1)
print("DRIVER RESULT: all probes passed")
