# Logging System Manual Test Guide

This guide helps you manually verify the logging system works correctly.

## Prerequisites

1. Backend running: `cd backend && source venv/bin/activate && uvicorn app.main:app --reload`
2. Frontend running: `cd frontend && nvm use 20 && npm run dev`
3. Terminal open for log monitoring: `tail -f ~/AddaxAI/logs/backend.log`

---

## Test 1: Backend Logging Basics

**Goal:** Verify backend logs are written correctly

### Steps:
1. Start the backend
2. Watch logs: `tail -f ~/AddaxAI/logs/backend.log`

### Expected Output:
```
[2024-12-15 10:00:00] [INFO] [addaxai] ================================================================================
[2024-12-15 10:00:00] [INFO] [addaxai] AddaxAI Backend Logging Initialized
[2024-12-15 10:00:00] [INFO] [addaxai] Log file: /Users/peter/AddaxAI/logs/backend.log
[2024-12-15 10:00:00] [INFO] [addaxai] Environment: development
[2024-12-15 10:00:00] [INFO] [addaxai] ================================================================================
[2024-12-15 10:00:00] [INFO] [addaxai.app.main] Starting AddaxAI Backend (Environment: development)
[2024-12-15 10:00:00] [INFO] [addaxai.app.main] Database: sqlite:///./addaxai.db
[2024-12-15 10:00:00] [INFO] [addaxai.app.db.base] Creating database tables...
[2024-12-15 10:00:00] [INFO] [addaxai.app.db.base] Database tables created successfully
```

✅ **Pass:** Log file exists and contains startup messages
❌ **Fail:** No log file or missing messages

---

## Test 2: API Operation Logging

**Goal:** Verify CRUD operations are logged

### Steps:
1. Open browser to http://localhost:5173
2. Click "New Project"
3. Create project with name "Test Logging Project"
4. Watch logs

### Expected Output:
```
[2024-12-15 10:05:00] [INFO] [addaxai.app.api.routers.projects] Created project: Test Logging Project (ID: abc123...)
```

### Additional Tests:
- **Update project:** Change description → Should log "Updated project: abc123..."
- **Delete project:** Delete → Should log "Deleted project: abc123... (cascaded to all related data)"
- **404 Error:** Try to get nonexistent project → Should log "[WARNING] Project not found: xyz..."

✅ **Pass:** All operations logged with correct level and context
❌ **Fail:** Operations not logged or wrong log level

---

## Test 3: Frontend Log Forwarding

**Goal:** Verify frontend logs are sent to backend

### Steps:
1. Open browser DevTools (F12)
2. Go to Console tab
3. Click around the app (navigate, click buttons, etc.)
4. Wait 5 seconds (for batch flush)
5. Check backend logs

### Expected Output:
```
[2024-12-15 10:10:00] [INFO] [addaxai.frontend] [2024-12-15T10:10:00.123Z] User clicked New Project button
[2024-12-15 10:10:01] [INFO] [addaxai.frontend] [2024-12-15T10:10:01.456Z] User navigated to project: Test Logging Project
```

### Frontend Console Output:
```
[INFO] User clicked New Project button
[Logger] Sent 5 logs to backend
```

✅ **Pass:** Frontend logs appear in backend.log with `[addaxai.frontend]` prefix
❌ **Fail:** Frontend logs don't appear or missing prefix

---

## Test 4: Error Logging

**Goal:** Verify errors are logged with stack traces

### Steps:
1. Try to create a site with invalid project_id:
   ```bash
   curl -X POST http://localhost:8000/api/sites \
     -H "Content-Type: application/json" \
     -d '{"project_id":"invalid","name":"Test","latitude":0,"longitude":0}'
   ```

2. Check logs

### Expected Output:
```
[2024-12-15 10:15:00] [WARNING] [addaxai.app.api.routers.sites] Failed to create site: project invalid not found
```

### Additional Error Tests:
- **Folder not found:** Create deployment with nonexistent folder → Should log `[ERROR]` with folder path
- **Duplicate name:** Create project with same name twice → Should log `[WARNING]` about duplicate

✅ **Pass:** All errors logged with appropriate level (ERROR/WARNING)
❌ **Fail:** Errors missing or logged at wrong level

---

## Test 5: Folder Scanner Logging

**Goal:** Verify folder scanning logs correctly (including warnings for corrupt files)

### Steps:
1. Create a test folder: `mkdir -p ~/test_deployment`
2. Create project and site via UI
3. Create deployment with folder_path = `~/test_deployment`
4. Trigger folder scan

### Expected Output:
```
[2024-12-15 10:20:00] [INFO] [addaxai.app.api.routers.deployments] Scanning folder for deployment abc123...: /Users/peter/test_deployment
[2024-12-15 10:20:00] [INFO] [addaxai.app.api.routers.deployments] Folder scan complete for abc123...: 0 images, 0 videos
```

### Test with Corrupt Files:
1. Create a fake image file: `echo "not an image" > ~/test_deployment/corrupt.jpg`
2. Rescan folder

### Expected:
```
[2024-12-15 10:21:00] [WARNING] [addaxai.app.services.folder_scanner] Failed to extract GPS from corrupt.jpg: ...
```

✅ **Pass:** Folder scans logged, warnings for corrupt files
❌ **Fail:** Silent failures (no logs for corrupt files)

---

## Test 6: Log File Rotation

**Goal:** Verify logs rotate correctly

### Steps:
1. Check current log file size: `ls -lh ~/AddaxAI/logs/backend.log`
2. If file is small, generate many logs (create/delete projects in loop)
3. Wait for file to reach 33MB
4. Verify backup created: `ls -lh ~/AddaxAI/logs/`

### Expected:
```
backend.log          (current log)
backend.log.1        (first backup)
backend.log.2        (second backup)
backend.log.3        (third backup)
```

✅ **Pass:** Log files rotate, max 4 total files
❌ **Fail:** File grows beyond 33MB or no backups created

---

## Test 7: Log Inspection Script

**Goal:** Verify log inspection tool works

### Steps:
1. Run: `python3 scripts/verify_logs.py`

### Expected Output:
```
================================================================================
📊 LOG ANALYSIS SUMMARY
================================================================================

✅ Total entries: 150
   - INFO: 140
   - WARNING: 8
   - ERROR: 2
   - CRITICAL: 0

📦 Log entries by module:
   - addaxai.app.main: 5
   - addaxai.app.api.routers.projects: 30
   ...

🔀 Log source:
   - Backend: 120
   - Frontend: 30

✅ No errors found

🔍 Key operations logged:
   ✅ Project creations: 5
   ✅ Site creations: 3
   ...

================================================================================
🏥 LOG HEALTH CHECK
================================================================================

✅ PASS: Found 150 log entries
✅ PASS: INFO logs present
✅ PASS: Backend logs present
✅ PASS: Frontend logs present
✅ PASS: No critical errors
✅ PASS: No silent failures detected
================================================================================

✅ Overall: Logging system appears healthy!
```

✅ **Pass:** Script runs and reports healthy
❌ **Fail:** Script errors or reports issues

---

## Common Issues

### Issue: No frontend logs appear
**Solution:** Wait 5 seconds for batch flush, or trigger error (errors flush immediately)

### Issue: Logs missing context
**Solution:** Check code uses `logger.info("msg", extra={...})` for context

### Issue: Silent failures
**Solution:** Check for `except Exception: pass` → should be `except Exception as e: logger.warning(...)`

---

## Quick Verification Checklist

Run through this checklist for a quick verification:

- [ ] Backend startup logged
- [ ] Create project logged
- [ ] Update project logged
- [ ] Delete project logged (with cascade message)
- [ ] 404 errors logged as WARNING
- [ ] Duplicate name errors logged as WARNING
- [ ] Frontend logs appear in backend.log
- [ ] Frontend logs have `[addaxai.frontend]` prefix
- [ ] Folder scan operations logged
- [ ] Corrupt file warnings logged
- [ ] Log file rotates at 33MB
- [ ] `verify_logs.py` script passes

If all checked ✅ - **Logging system is working correctly!**
