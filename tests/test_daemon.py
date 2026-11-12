"""Tests for responsive daemon control and queued requests."""

import json
import socket

import pytest

from bts import cli
from bts.daemon import BTSDaemon


def dispatch(daemon, request):
    server, client = socket.socketpair()
    try:
        owns_connection = daemon._dispatch(request, server)
        reader = client.makefile("r")
        try:
            response = json.loads(reader.readline())
        finally:
            reader.close()
        return owns_connection, response, server, client
    except Exception:
        server.close()
        client.close()
        raise


def enqueue(daemon, *requests):
    pairs = []
    with daemon._queue_condition:
        for request in requests:
            server, client = socket.socketpair()
            daemon._work_queue.append((request, server))
            pairs.append((server, client))
    return pairs


def task_texts(daemon):
    return [request["text"] for request, _ in daemon._work_queue]


def run_command(daemon, request):
    owns_connection, response, server, client = dispatch(daemon, request)
    server.close()
    client.close()
    assert not owns_connection
    return response


def test_status_responds_while_a_request_is_active():
    daemon = BTSDaemon()
    daemon._active_action = "play"

    owns_connection, response, server, client = dispatch(
        daemon, {"action": "status"}
    )
    try:
        assert not owns_connection
        assert response["status"] == "ok"
        assert "processing" in response["message"]
        assert "current request: play" in response["message"]
    finally:
        server.close()
        client.close()


def test_synthesis_requests_are_queued():
    daemon = BTSDaemon()
    owns_connection, response, server, client = dispatch(
        daemon, {"action": "play", "text": "hola"}
    )
    try:
        assert owns_connection
        assert response["status"] == "progress"
        assert "queued" in response["message"]
        assert len(daemon._work_queue) == 1
    finally:
        server.close()
        client.close()


def test_play_handler_resolves_auto_device(monkeypatch):
    from contextlib import nullcontext

    import bts.device
    import bts.synthesis

    daemon = BTSDaemon()
    captured = {}

    def resolve_device(requested):
        captured["requested_device"] = requested
        return "cpu"

    def synthesize(text, **kwargs):
        captured["text"] = text
        captured.update(kwargs)

    monkeypatch.setattr(bts.device, "resolve_device", resolve_device)
    monkeypatch.setattr(bts.synthesis, "synthesize_and_play", synthesize)
    monkeypatch.setattr(daemon, "_capture_stdout", lambda conn: nullcontext())

    server, client = socket.socketpair()
    try:
        daemon._handle_play({"text": "hola"}, server)
        response = json.loads(client.recv(4096))
        assert response["status"] == "ok"
        assert captured["requested_device"] == "auto"
        assert captured["device"] == "cpu"
    finally:
        server.close()
        client.close()


def test_play_and_resume_only_resume_a_paused_conversion():
    daemon = BTSDaemon()
    _, started, server, client = dispatch(daemon, {"action": "start"})
    assert started["message"] == "Daemon is already running."
    server.close()
    client.close()

    _, ignored, server, client = dispatch(daemon, {"action": "play"})
    assert ignored["message"] == "No paused processing to resume."
    server.close()
    client.close()

    _, paused, server, client = dispatch(daemon, {"action": "pause"})
    assert paused["message"] == "Processing paused."
    server.close()
    client.close()

    _, resumed, server, client = dispatch(daemon, {"action": "resume"})
    try:
        assert resumed["message"] == "Processing resumed."
        assert daemon._resume_event.is_set()
    finally:
        server.close()
        client.close()

    _, paused, server, client = dispatch(daemon, {"action": "pause"})
    assert paused["message"] == "Processing paused."
    server.close()
    client.close()

    _, played, server, client = dispatch(daemon, {"action": "play"})
    try:
        assert played["message"] == "Processing resumed."
        assert daemon._resume_event.is_set()
    finally:
        server.close()
        client.close()


def test_stop_cancels_the_active_request():
    daemon = BTSDaemon()
    daemon._active_action = "save"
    _, response, server, client = dispatch(daemon, {"action": "stop"})
    try:
        assert response["message"] == "Stopping current request."
        assert daemon._cancel_current.is_set()
    finally:
        server.close()
        client.close()


def test_next_skips_active_task_and_resumes_queue():
    daemon = BTSDaemon()
    daemon._active_action = "play"
    daemon._active_request = {"action": "play", "text": "running"}
    daemon._paused = True
    daemon._resume_event.clear()
    pairs = enqueue(daemon, {"action": "play", "text": "next"})

    try:
        _, response, server, client = dispatch(daemon, {"action": "next"})
        try:
            assert response["message"] == (
                "Current task skipped; moving to the next task."
            )
            assert daemon._cancel_current.is_set()
            assert not daemon._paused
            assert daemon._resume_event.is_set()
            assert task_texts(daemon) == ["next"]
        finally:
            server.close()
            client.close()
    finally:
        for server, client in pairs:
            server.close()
            client.close()


def test_list_includes_active_and_pending_tasks():
    daemon = BTSDaemon()
    daemon._active_action = "play"
    daemon._active_request = {"action": "play", "text": "speaking"}
    pairs = enqueue(
        daemon,
        {"action": "play", "text": "first"},
        {"action": "play", "text": "second"},
    )
    _, response, server, client = dispatch(daemon, {"action": "list"})
    try:
        assert response["status"] == "ok"
        assert "1. [running] play: speaking" in response["message"]
        assert "2. [pending] play: first" in response["message"]
        assert "3. [pending] play: second" in response["message"]
    finally:
        server.close()
        client.close()
        for pair in pairs:
            pair[0].close()
            pair[1].close()


def test_up_down_and_move_reorder_pending_tasks():
    daemon = BTSDaemon()
    pairs = enqueue(
        daemon,
        {"action": "play", "text": "a"},
        {"action": "play", "text": "b"},
        {"action": "play", "text": "c"},
    )
    try:
        run_command(daemon, {"action": "up", "number": 3})
        assert task_texts(daemon) == ["a", "c", "b"]
        run_command(daemon, {"action": "down", "number": 1})
        assert task_texts(daemon) == ["c", "a", "b"]
        run_command(daemon, {"action": "move", "from": 2, "to": 999})
        assert task_texts(daemon) == ["c", "b", "a"]
        run_command(daemon, {"action": "move", "from": 3, "to": 1})
        assert task_texts(daemon) == ["a", "c", "b"]
    finally:
        for server, client in pairs:
            server.close()
            client.close()


def test_move_to_running_task_places_pending_task_next():
    daemon = BTSDaemon()
    daemon._active_action = "play"
    daemon._active_request = {"action": "play", "text": "running"}
    pairs = enqueue(
        daemon,
        {"action": "play", "text": "first"},
        {"action": "play", "text": "second"},
    )
    try:
        owns_connection, response, server, client = dispatch(
            daemon, {"action": "move", "from": 3, "to": 1}
        )
        try:
            assert not owns_connection
            assert response["message"] == "Task 3 moved to position 2."
            assert task_texts(daemon) == ["second", "first"]
        finally:
            server.close()
            client.close()
    finally:
        for server, client in pairs:
            server.close()
            client.close()


def test_delete_removes_pending_task_and_completes_its_client():
    daemon = BTSDaemon()
    pairs = enqueue(
        daemon,
        {"action": "play", "text": "remove"},
        {"action": "play", "text": "keep"},
    )
    try:
        owns_connection, response, server, client = dispatch(
            daemon, {"action": "delete", "number": 1}
        )
        try:
            assert not owns_connection
            assert response["message"] == "Task 1 deleted."
            assert task_texts(daemon) == ["keep"]
            removed = json.loads(pairs[0][1].makefile("r").readline())
            assert removed["status"] == "error"
            assert removed["done"] is True
        finally:
            server.close()
            client.close()
    finally:
        for server, client in pairs:
            server.close()
            client.close()


def test_running_task_cannot_be_deleted_or_moved():
    daemon = BTSDaemon()
    daemon._active_action = "play"
    daemon._active_request = {"action": "play", "text": "running"}
    for request in (
        {"action": "delete", "number": 1},
        {"action": "move", "from": 1, "to": 2},
    ):
        _, response, server, client = dispatch(daemon, request)
        try:
            assert response["status"] == "error"
            assert "running task cannot" in response["message"]
        finally:
            server.close()
            client.close()


def test_cli_exposes_queue_commands(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli, "_daemon_control",
        lambda action, **params: calls.append((action, params)) or 0,
    )
    assert cli._handle_daemon_command(["list"]) == 0
    assert cli._handle_daemon_command(["delete", "2"]) == 0
    assert cli._handle_daemon_command(["up", "3"]) == 0
    assert cli._handle_daemon_command(["down", "4"]) == 0
    assert cli._handle_daemon_command(["move", "5", "2"]) == 0
    assert cli._handle_daemon_command(["next"]) == 0
    assert cli._handle_daemon_command(["play"]) == 0
    assert cli._handle_daemon_command(["resume"]) == 0
    assert calls == [
        ("list", {}),
        ("delete", {"number": 2}),
        ("up", {"number": 3}),
        ("down", {"number": 4}),
        ("move", {"from": 5, "to": 2}),
        ("next", {}),
        ("play", {}),
        ("resume", {}),
    ]


def test_daemon_help_is_available_with_or_without_help_command(capsys):
    assert cli._handle_daemon_command(["help"]) == 0
    explicit_help = capsys.readouterr().out
    assert "list" in explicit_help
    assert "resume" in explicit_help
    assert "next" in explicit_help

    assert cli._handle_daemon_command([]) == 0
    implicit_help = capsys.readouterr().out
    assert "list" in implicit_help
    assert "resume" in implicit_help
    assert "next" in implicit_help


def test_main_help_lists_daemon_commands(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.build_parser().parse_args(["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "Daemon commands:" in help_text
    assert "bts daemon next" in help_text
