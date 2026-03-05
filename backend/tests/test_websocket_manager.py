"""
Tests for the WebSocket ConnectionManager.

Covers the ready-handshake protocol, state management, reconnection behavior,
and edge cases around concurrent connections and cleanup.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.websocket_manager import ConnectionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_ws(*, accept_side_effect=None) -> AsyncMock:
    """Create a mock WebSocket with send_json and accept methods."""
    ws = AsyncMock()
    ws.accept = AsyncMock(side_effect=accept_side_effect)
    ws.send_json = AsyncMock()
    return ws


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------

class TestConnect:
    async def test_connect_accepts_and_registers(self):
        mgr = ConnectionManager()
        ws = make_mock_ws()

        await mgr.connect(ws, "job-1")

        ws.accept.assert_awaited_once()
        assert ws in mgr.active_connections["job-1"]

    async def test_connect_multiple_clients_same_job(self):
        mgr = ConnectionManager()
        ws1 = make_mock_ws()
        ws2 = make_mock_ws()

        await mgr.connect(ws1, "job-1")
        await mgr.connect(ws2, "job-1")

        assert len(mgr.active_connections["job-1"]) == 2
        assert ws1 in mgr.active_connections["job-1"]
        assert ws2 in mgr.active_connections["job-1"]

    async def test_connect_sends_current_state_on_reconnect(self):
        mgr = ConnectionManager()
        ws = make_mock_ws()

        # Simulate existing state (job in progress)
        mgr.current_state["job-1"] = {
            "type": "progress",
            "job_id": "job-1",
            "message": "Processing...",
            "progress": 0.5,
        }

        await mgr.connect(ws, "job-1")

        # Should send the current state to catch up the reconnecting client
        ws.send_json.assert_awaited_once_with(mgr.current_state["job-1"])

    async def test_connect_no_state_sends_nothing(self):
        mgr = ConnectionManager()
        ws = make_mock_ws()

        await mgr.connect(ws, "job-1")

        # No current state → no send_json call
        ws.send_json.assert_not_awaited()

    async def test_connect_sends_complete_state_on_reconnect(self):
        """Reconnecting after completion should send the complete message."""
        mgr = ConnectionManager()
        ws = make_mock_ws()

        mgr.current_state["job-1"] = {
            "type": "complete",
            "job_id": "job-1",
            "success": True,
            "message": "Done",
        }

        await mgr.connect(ws, "job-1")

        ws.send_json.assert_awaited_once()
        sent = ws.send_json.call_args[0][0]
        assert sent["type"] == "complete"

    async def test_connect_different_jobs_isolated(self):
        """Connections for different jobs don't interfere."""
        mgr = ConnectionManager()
        ws_a = make_mock_ws()
        ws_b = make_mock_ws()

        mgr.current_state["job-a"] = {
            "type": "progress",
            "job_id": "job-a",
            "message": "A running",
            "progress": 0.3,
        }

        await mgr.connect(ws_a, "job-a")
        await mgr.connect(ws_b, "job-b")

        # ws_a gets state for job-a
        ws_a.send_json.assert_awaited_once()
        # ws_b gets nothing (no state for job-b)
        ws_b.send_json.assert_not_awaited()


class TestDisconnect:
    async def test_disconnect_removes_connection(self):
        mgr = ConnectionManager()
        ws = make_mock_ws()

        await mgr.connect(ws, "job-1")
        assert mgr.get_connection_count("job-1") == 1

        await mgr.disconnect(ws, "job-1")
        assert mgr.get_connection_count("job-1") == 0
        assert "job-1" not in mgr.active_connections

    async def test_disconnect_keeps_other_connections(self):
        mgr = ConnectionManager()
        ws1 = make_mock_ws()
        ws2 = make_mock_ws()

        await mgr.connect(ws1, "job-1")
        await mgr.connect(ws2, "job-1")

        await mgr.disconnect(ws1, "job-1")
        assert mgr.get_connection_count("job-1") == 1
        assert ws2 in mgr.active_connections["job-1"]


# ---------------------------------------------------------------------------
# Ready-handshake protocol
# ---------------------------------------------------------------------------

class TestReadyHandshake:
    async def test_register_start_stores_function(self):
        mgr = ConnectionManager()
        fn = AsyncMock()

        mgr.register_start("job-1", fn)

        assert "job-1" in mgr._pending_starts

    async def test_handle_ready_starts_worker(self):
        mgr = ConnectionManager()
        started = asyncio.Event()

        async def worker():
            started.set()

        mgr.register_start("job-1", worker)
        await mgr.handle_ready("job-1")

        # Give the created task a chance to run
        await asyncio.sleep(0.05)
        assert started.is_set()

    async def test_handle_ready_pops_start_fn(self):
        """After handle_ready, the start function is removed."""
        mgr = ConnectionManager()
        mgr.register_start("job-1", AsyncMock())

        await mgr.handle_ready("job-1")
        await asyncio.sleep(0.01)

        assert "job-1" not in mgr._pending_starts

    async def test_handle_ready_idempotent(self):
        """Second 'ready' for the same job is a no-op (reconnection case)."""
        mgr = ConnectionManager()
        call_count = 0

        async def worker():
            nonlocal call_count
            call_count += 1

        mgr.register_start("job-1", worker)

        await mgr.handle_ready("job-1")
        await asyncio.sleep(0.05)
        assert call_count == 1

        # Second ready — should NOT start another worker
        await mgr.handle_ready("job-1")
        await asyncio.sleep(0.05)
        assert call_count == 1

    async def test_handle_ready_no_registration(self):
        """handle_ready with no registered start is a safe no-op."""
        mgr = ConnectionManager()
        # Should not raise
        await mgr.handle_ready("nonexistent-job")

    async def test_register_start_overwrites_previous(self):
        """Registering start for same job_id replaces previous."""
        mgr = ConnectionManager()
        first = asyncio.Event()
        second = asyncio.Event()

        async def worker1():
            first.set()

        async def worker2():
            second.set()

        mgr.register_start("job-1", worker1)
        mgr.register_start("job-1", worker2)

        await mgr.handle_ready("job-1")
        await asyncio.sleep(0.05)

        assert not first.is_set()
        assert second.is_set()

    async def test_multiple_jobs_independent(self):
        """Ready signals for different jobs start independent workers."""
        mgr = ConnectionManager()
        started_a = asyncio.Event()
        started_b = asyncio.Event()

        async def worker_a():
            started_a.set()

        async def worker_b():
            started_b.set()

        mgr.register_start("job-a", worker_a)
        mgr.register_start("job-b", worker_b)

        await mgr.handle_ready("job-a")
        await asyncio.sleep(0.05)

        assert started_a.is_set()
        assert not started_b.is_set()  # job-b not started yet

        await mgr.handle_ready("job-b")
        await asyncio.sleep(0.05)

        assert started_b.is_set()


# ---------------------------------------------------------------------------
# Sending messages
# ---------------------------------------------------------------------------

class TestSendProgress:
    async def test_send_progress_to_connected_client(self):
        mgr = ConnectionManager()
        ws = make_mock_ws()

        await mgr.connect(ws, "job-1")
        ws.send_json.reset_mock()  # Clear the connect call

        await mgr.send_progress("job-1", "Working...", 0.5, "detection", 0.3)

        ws.send_json.assert_awaited_once()
        sent = ws.send_json.call_args[0][0]
        assert sent["type"] == "progress"
        assert sent["job_id"] == "job-1"
        assert sent["message"] == "Working..."
        assert sent["progress"] == 0.5
        assert sent["phase"] == "detection"
        assert sent["phase_progress"] == 0.3

    async def test_send_progress_stores_current_state(self):
        mgr = ConnectionManager()

        await mgr.send_progress("job-1", "Working...", 0.5)

        assert "job-1" in mgr.current_state
        assert mgr.current_state["job-1"]["type"] == "progress"
        assert mgr.current_state["job-1"]["progress"] == 0.5

    async def test_send_progress_replaces_previous_state(self):
        mgr = ConnectionManager()

        await mgr.send_progress("job-1", "Step 1", 0.3)
        await mgr.send_progress("job-1", "Step 2", 0.6)

        assert mgr.current_state["job-1"]["message"] == "Step 2"
        assert mgr.current_state["job-1"]["progress"] == 0.6

    async def test_send_progress_to_multiple_clients(self):
        mgr = ConnectionManager()
        ws1 = make_mock_ws()
        ws2 = make_mock_ws()

        await mgr.connect(ws1, "job-1")
        await mgr.connect(ws2, "job-1")
        ws1.send_json.reset_mock()
        ws2.send_json.reset_mock()

        await mgr.send_progress("job-1", "Working...", 0.5)

        ws1.send_json.assert_awaited_once()
        ws2.send_json.assert_awaited_once()

    async def test_send_progress_no_connections_still_stores_state(self):
        """Progress sent with no connected clients still stores state."""
        mgr = ConnectionManager()

        await mgr.send_progress("job-1", "Working...", 0.5)

        assert mgr.current_state["job-1"]["progress"] == 0.5

    async def test_send_progress_removes_failed_connections(self):
        mgr = ConnectionManager()
        good_ws = make_mock_ws()
        bad_ws = make_mock_ws()
        bad_ws.send_json = AsyncMock(side_effect=Exception("Connection lost"))

        await mgr.connect(good_ws, "job-1")
        await mgr.connect(bad_ws, "job-1")
        assert mgr.get_connection_count("job-1") == 2

        await mgr.send_progress("job-1", "Working...", 0.5)

        # Bad connection should be removed
        assert mgr.get_connection_count("job-1") == 1
        assert good_ws in mgr.active_connections["job-1"]

    async def test_send_progress_includes_data(self):
        mgr = ConnectionManager()
        ws = make_mock_ws()

        await mgr.connect(ws, "job-1")
        ws.send_json.reset_mock()

        await mgr.send_progress(
            "job-1", "Detecting...", 0.5,
            data={"deployment_index": 1, "total_deployments": 3}
        )

        sent = ws.send_json.call_args[0][0]
        assert sent["data"]["deployment_index"] == 1
        assert sent["data"]["total_deployments"] == 3

    async def test_send_progress_different_jobs_isolated(self):
        """Progress for job-a doesn't reach job-b clients."""
        mgr = ConnectionManager()
        ws_a = make_mock_ws()
        ws_b = make_mock_ws()

        await mgr.connect(ws_a, "job-a")
        await mgr.connect(ws_b, "job-b")
        ws_a.send_json.reset_mock()
        ws_b.send_json.reset_mock()

        await mgr.send_progress("job-a", "Working...", 0.5)

        ws_a.send_json.assert_awaited_once()
        ws_b.send_json.assert_not_awaited()


class TestSendComplete:
    async def test_send_complete_to_client(self):
        mgr = ConnectionManager()
        ws = make_mock_ws()

        await mgr.connect(ws, "job-1")
        ws.send_json.reset_mock()

        await mgr.send_complete("job-1", success=True, message="All done")

        ws.send_json.assert_awaited_once()
        sent = ws.send_json.call_args[0][0]
        assert sent["type"] == "complete"
        assert sent["success"] is True
        assert sent["message"] == "All done"

    async def test_send_complete_stores_state(self):
        mgr = ConnectionManager()

        await mgr.send_complete("job-1", success=True, message="Done")

        assert mgr.current_state["job-1"]["type"] == "complete"

    async def test_send_complete_overwrites_progress_state(self):
        mgr = ConnectionManager()

        await mgr.send_progress("job-1", "Working...", 0.5)
        assert mgr.current_state["job-1"]["type"] == "progress"

        await mgr.send_complete("job-1", success=True, message="Done")
        assert mgr.current_state["job-1"]["type"] == "complete"

    async def test_send_complete_schedules_cleanup(self):
        """Complete state should be cleaned up after delay."""
        mgr = ConnectionManager()

        with patch.object(mgr, "_cleanup_state", new_callable=AsyncMock) as mock_cleanup:
            # Patch asyncio.create_task to capture the coroutine
            original_create_task = asyncio.create_task

            created_tasks = []
            def capture_create_task(coro):
                task = original_create_task(coro)
                created_tasks.append(task)
                return task

            with patch("asyncio.create_task", side_effect=capture_create_task):
                await mgr.send_complete("job-1", success=True, message="Done")

            # A cleanup task should have been created
            assert len(created_tasks) >= 1

    async def test_send_complete_with_data(self):
        mgr = ConnectionManager()
        ws = make_mock_ws()

        await mgr.connect(ws, "job-1")
        ws.send_json.reset_mock()

        await mgr.send_complete(
            "job-1", success=True, message="Done",
            data={"total_detections": 42}
        )

        sent = ws.send_json.call_args[0][0]
        assert sent["data"]["total_detections"] == 42


class TestSendError:
    async def test_send_error_to_client(self):
        mgr = ConnectionManager()
        ws = make_mock_ws()

        await mgr.connect(ws, "job-1")
        ws.send_json.reset_mock()

        await mgr.send_error("job-1", "Something broke")

        ws.send_json.assert_awaited_once()
        sent = ws.send_json.call_args[0][0]
        assert sent["type"] == "error"
        assert sent["message"] == "Something broke"

    async def test_send_error_stores_state(self):
        mgr = ConnectionManager()

        await mgr.send_error("job-1", "Boom")

        assert mgr.current_state["job-1"]["type"] == "error"

    async def test_send_error_overwrites_progress_state(self):
        mgr = ConnectionManager()

        await mgr.send_progress("job-1", "Working...", 0.5)
        await mgr.send_error("job-1", "Failed")

        assert mgr.current_state["job-1"]["type"] == "error"


# ---------------------------------------------------------------------------
# State cleanup
# ---------------------------------------------------------------------------

class TestCleanupState:
    async def test_cleanup_removes_state_after_delay(self):
        mgr = ConnectionManager()
        mgr.current_state["job-1"] = {"type": "complete"}

        await mgr._cleanup_state("job-1", delay=0)

        assert "job-1" not in mgr.current_state

    async def test_cleanup_nonexistent_job_is_noop(self):
        mgr = ConnectionManager()

        # Should not raise
        await mgr._cleanup_state("nonexistent", delay=0)

    async def test_cleanup_preserves_other_jobs(self):
        mgr = ConnectionManager()
        mgr.current_state["job-1"] = {"type": "complete"}
        mgr.current_state["job-2"] = {"type": "progress"}

        await mgr._cleanup_state("job-1", delay=0)

        assert "job-1" not in mgr.current_state
        assert "job-2" in mgr.current_state


class TestCleanupPendingStart:
    async def test_cleanup_removes_orphaned_pending_start(self):
        """Pending start is removed after timeout if no 'ready' received."""
        mgr = ConnectionManager()
        mgr._pending_starts["job-1"] = AsyncMock()

        await mgr._cleanup_pending_start("job-1", delay=0)

        assert "job-1" not in mgr._pending_starts

    async def test_cleanup_noop_if_already_started(self):
        """If 'ready' already popped the start_fn, cleanup is a no-op."""
        mgr = ConnectionManager()
        mgr._pending_starts["job-1"] = AsyncMock()

        # Simulate "ready" consuming the start function
        await mgr.handle_ready("job-1")
        await asyncio.sleep(0.05)

        # Cleanup should not raise
        await mgr._cleanup_pending_start("job-1", delay=0)
        assert "job-1" not in mgr._pending_starts

    async def test_cleanup_does_not_affect_other_tasks(self):
        """Cleaning up one task doesn't affect another."""
        mgr = ConnectionManager()
        mgr._pending_starts["job-1"] = AsyncMock()
        mgr._pending_starts["job-2"] = AsyncMock()

        await mgr._cleanup_pending_start("job-1", delay=0)

        assert "job-1" not in mgr._pending_starts
        assert "job-2" in mgr._pending_starts


# ---------------------------------------------------------------------------
# Connection count
# ---------------------------------------------------------------------------

class TestGetConnectionCount:
    async def test_zero_for_unknown_job(self):
        mgr = ConnectionManager()
        assert mgr.get_connection_count("nonexistent") == 0

    async def test_counts_active_connections(self):
        mgr = ConnectionManager()
        ws1 = make_mock_ws()
        ws2 = make_mock_ws()

        await mgr.connect(ws1, "job-1")
        assert mgr.get_connection_count("job-1") == 1

        await mgr.connect(ws2, "job-1")
        assert mgr.get_connection_count("job-1") == 2


# ---------------------------------------------------------------------------
# Full flow integration tests
# ---------------------------------------------------------------------------

class TestFullFlow:
    async def test_normal_job_lifecycle(self):
        """Simulate: register → connect → ready → progress → complete."""
        mgr = ConnectionManager()
        ws = make_mock_ws()
        work_done = asyncio.Event()

        async def worker():
            await mgr.send_progress("job-1", "Step 1", 0.5)
            await mgr.send_progress("job-1", "Step 2", 0.9)
            await mgr.send_complete("job-1", success=True, message="All done")
            work_done.set()

        # 1. Register worker
        mgr.register_start("job-1", worker)

        # 2. Client connects
        await mgr.connect(ws, "job-1")
        ws.send_json.reset_mock()

        # 3. Client sends ready
        await mgr.handle_ready("job-1")

        # 4. Wait for worker to finish
        await asyncio.wait_for(work_done.wait(), timeout=2.0)

        # Verify messages received
        calls = [c[0][0] for c in ws.send_json.call_args_list]
        assert len(calls) == 3
        assert calls[0]["type"] == "progress"
        assert calls[0]["message"] == "Step 1"
        assert calls[1]["type"] == "progress"
        assert calls[1]["message"] == "Step 2"
        assert calls[2]["type"] == "complete"

        # State should be "complete"
        assert mgr.current_state["job-1"]["type"] == "complete"

    async def test_reconnect_receives_current_state(self):
        """Client disconnects mid-job and reconnects → gets latest state."""
        mgr = ConnectionManager()
        ws1 = make_mock_ws()
        ws2 = make_mock_ws()

        # Simulate a job in progress
        await mgr.connect(ws1, "job-1")
        await mgr.send_progress("job-1", "Processing...", 0.6, "detection", 0.8)

        # Client disconnects
        await mgr.disconnect(ws1, "job-1")

        # Client reconnects
        await mgr.connect(ws2, "job-1")

        # New client should receive the latest state
        ws2.send_json.assert_awaited_once()
        sent = ws2.send_json.call_args[0][0]
        assert sent["type"] == "progress"
        assert sent["progress"] == 0.6

    async def test_reconnect_after_complete(self):
        """Client reconnects after job completed → gets complete message."""
        mgr = ConnectionManager()
        ws1 = make_mock_ws()

        # Job completes with no active connections
        await mgr.send_complete("job-1", success=True, message="Done")

        # Client connects (late)
        await mgr.connect(ws1, "job-1")

        ws1.send_json.assert_awaited_once()
        sent = ws1.send_json.call_args[0][0]
        assert sent["type"] == "complete"
        assert sent["message"] == "Done"

    async def test_ready_after_reconnect_does_not_restart(self):
        """Reconnecting client sends 'ready' again → no duplicate worker."""
        mgr = ConnectionManager()
        call_count = 0

        async def worker():
            nonlocal call_count
            call_count += 1

        mgr.register_start("job-1", worker)

        # First ready → starts worker
        await mgr.handle_ready("job-1")
        await asyncio.sleep(0.05)
        assert call_count == 1

        # Client reconnects and sends ready again → no-op
        await mgr.handle_ready("job-1")
        await asyncio.sleep(0.05)
        assert call_count == 1

    async def test_sequential_jobs_no_cross_contamination(self):
        """Two sequential jobs don't interfere with each other's state."""
        mgr = ConnectionManager()

        # Job A completes
        ws_a = make_mock_ws()
        await mgr.connect(ws_a, "job-a")

        completed_a = asyncio.Event()
        async def worker_a():
            await mgr.send_progress("job-a", "Working A", 0.5)
            await mgr.send_complete("job-a", success=True, message="A done")
            completed_a.set()

        mgr.register_start("job-a", worker_a)
        await mgr.handle_ready("job-a")
        await asyncio.wait_for(completed_a.wait(), timeout=2.0)

        assert mgr.current_state["job-a"]["type"] == "complete"

        # Job B starts — should NOT see job A's state
        ws_b = make_mock_ws()
        await mgr.connect(ws_b, "job-b")

        # ws_b should NOT receive job-a's complete message
        ws_b.send_json.assert_not_awaited()

        # Job B's state is independent
        assert "job-b" not in mgr.current_state

        completed_b = asyncio.Event()
        async def worker_b():
            await mgr.send_progress("job-b", "Working B", 0.3)
            completed_b.set()

        mgr.register_start("job-b", worker_b)
        await mgr.handle_ready("job-b")
        await asyncio.wait_for(completed_b.wait(), timeout=2.0)

        # Both states exist independently
        assert mgr.current_state["job-a"]["type"] == "complete"
        assert mgr.current_state["job-b"]["type"] == "progress"

    async def test_worker_error_sends_error_message(self):
        """Worker that raises → error sent to client."""
        mgr = ConnectionManager()
        ws = make_mock_ws()

        await mgr.connect(ws, "job-1")
        ws.send_json.reset_mock()

        async def failing_worker():
            await mgr.send_progress("job-1", "Starting...", 0.1)
            # Simulate error handling (worker catches and sends error)
            await mgr.send_error("job-1", "Pipeline crashed")

        mgr.register_start("job-1", failing_worker)
        await mgr.handle_ready("job-1")
        await asyncio.sleep(0.1)

        calls = [c[0][0] for c in ws.send_json.call_args_list]
        assert len(calls) == 2
        assert calls[0]["type"] == "progress"
        assert calls[1]["type"] == "error"
        assert mgr.current_state["job-1"]["type"] == "error"

    async def test_concurrent_jobs_isolated(self):
        """Two jobs running concurrently don't interfere."""
        mgr = ConnectionManager()
        ws_a = make_mock_ws()
        ws_b = make_mock_ws()

        await mgr.connect(ws_a, "job-a")
        await mgr.connect(ws_b, "job-b")
        ws_a.send_json.reset_mock()
        ws_b.send_json.reset_mock()

        done_a = asyncio.Event()
        done_b = asyncio.Event()

        async def worker_a():
            await mgr.send_progress("job-a", "A step 1", 0.5)
            await asyncio.sleep(0.05)
            await mgr.send_complete("job-a", success=True, message="A done")
            done_a.set()

        async def worker_b():
            await mgr.send_progress("job-b", "B step 1", 0.3)
            await asyncio.sleep(0.05)
            await mgr.send_complete("job-b", success=True, message="B done")
            done_b.set()

        mgr.register_start("job-a", worker_a)
        mgr.register_start("job-b", worker_b)

        await mgr.handle_ready("job-a")
        await mgr.handle_ready("job-b")

        await asyncio.wait_for(done_a.wait(), timeout=2.0)
        await asyncio.wait_for(done_b.wait(), timeout=2.0)

        # Verify ws_a only got job-a messages
        a_calls = [c[0][0] for c in ws_a.send_json.call_args_list]
        assert all(m["job_id"] == "job-a" for m in a_calls)

        # Verify ws_b only got job-b messages
        b_calls = [c[0][0] for c in ws_b.send_json.call_args_list]
        assert all(m["job_id"] == "job-b" for m in b_calls)
