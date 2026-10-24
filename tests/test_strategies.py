"""Tests for Cozempic strategies."""

from __future__ import annotations

import json
import unittest

from cozempic.helpers import msg_bytes
from cozempic.registry import STRATEGIES

# Ensure strategies are registered
import cozempic.strategies  # noqa: F401


def make_message(line_idx: int, msg: dict) -> tuple[int, dict, int]:
    return (line_idx, msg, msg_bytes(msg))


def make_progress(line_idx: int) -> tuple[int, dict, int]:
    msg = {
        "type": "progress",
        "data": {"type": "hook_progress"},
        "timestamp": "2026-01-01T00:00:00Z",
        "uuid": f"uuid-{line_idx}",
    }
    return make_message(line_idx, msg)


def make_assistant(line_idx: int, text: str = "hello", thinking: str | None = None) -> tuple[int, dict, int]:
    content = []
    if thinking:
        content.append({"type": "thinking", "thinking": thinking, "signature": "sig" * 50})
    content.append({"type": "text", "text": text})
    msg = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": content,
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "stop_reason": "end_turn",
        },
        "costUSD": 0.01,
        "duration": 1234,
    }
    return make_message(line_idx, msg)


def make_user(line_idx: int, text: str = "hi") -> tuple[int, dict, int]:
    msg = {
        "type": "user",
        "cwd": "/home/user/project",
        "version": "2.1.25",
        "slug": "test-slug",
        "gitBranch": "main",
        "userType": "external",
        "isSidechain": False,
        "message": {"role": "user", "content": text},
        "uuid": f"uuid-{line_idx}",
    }
    return make_message(line_idx, msg)


def make_file_history(line_idx: int, mid: str = "mid-1") -> tuple[int, dict, int]:
    msg = {
        "type": "file-history-snapshot",
        "messageId": mid,
        "snapshot": {"messageId": mid, "trackedFileBackups": {}, "timestamp": "2026-01-01T00:00:00Z"},
        "isSnapshotUpdate": False,
    }
    return make_message(line_idx, msg)


class TestProgressCollapse(unittest.TestCase):
    def test_collapses_consecutive_progress(self):
        messages = [
            make_progress(0),
            make_progress(1),
            make_progress(2),
            make_user(3),
            make_progress(4),
        ]
        sr = STRATEGIES["progress-collapse"].func(messages, {})
        # Run of 3: remove 0,1 keep 2. Isolated 4: also removed. Total=3
        self.assertEqual(sr.messages_removed, 3)
        self.assertEqual(sr.messages_replaced, 0)
        removed_lines = {a.line_index for a in sr.actions}
        self.assertEqual(removed_lines, {0, 1, 4})

    def test_single_progress_removed(self):
        """Isolated progress ticks are now removed (not kept)."""
        messages = [make_user(0), make_progress(1), make_user(2)]
        sr = STRATEGIES["progress-collapse"].func(messages, {})
        self.assertEqual(len(sr.actions), 1)
        self.assertEqual(sr.actions[0].line_index, 1)


class TestFileHistoryDedup(unittest.TestCase):
    def test_dedup_same_message_id(self):
        messages = [
            make_file_history(0, "mid-1"),
            make_file_history(1, "mid-1"),
            make_file_history(2, "mid-1"),
        ]
        sr = STRATEGIES["file-history-dedup"].func(messages, {})
        self.assertEqual(sr.messages_removed, 2)
        # Should keep last (line 2), remove lines 0 and 1
        removed_lines = {a.line_index for a in sr.actions}
        self.assertEqual(removed_lines, {0, 1})

    def test_different_ids_not_deduped(self):
        messages = [
            make_file_history(0, "mid-1"),
            make_file_history(1, "mid-2"),
        ]
        sr = STRATEGIES["file-history-dedup"].func(messages, {})
        self.assertEqual(len(sr.actions), 0)


class TestMetadataStrip(unittest.TestCase):
    def test_strips_usage_and_costs(self):
        messages = [make_assistant(0, text="hi", thinking="hmm")]
        sr = STRATEGIES["metadata-strip"].func(messages, {})
        self.assertEqual(sr.messages_replaced, 1)
        replacement = sr.actions[0].replacement
        self.assertNotIn("costUSD", replacement)
        self.assertNotIn("duration", replacement)
        self.assertNotIn("usage", replacement.get("message", {}))
        self.assertNotIn("stop_reason", replacement.get("message", {}))


class TestThinkingBlocks(unittest.TestCase):
    def test_remove_mode(self):
        messages = [make_assistant(0, text="hi", thinking="deep thoughts")]
        sr = STRATEGIES["thinking-blocks"].func(messages, {"thinking_mode": "remove"})
        self.assertEqual(sr.messages_replaced, 1)
        replacement = sr.actions[0].replacement
        content = replacement["message"]["content"]
        types = [b["type"] for b in content]
        self.assertNotIn("thinking", types)

    def test_truncate_mode(self):
        long_thinking = "x" * 500
        messages = [make_assistant(0, text="hi", thinking=long_thinking)]
        sr = STRATEGIES["thinking-blocks"].func(messages, {"thinking_mode": "truncate"})
        self.assertEqual(sr.messages_replaced, 1)
        replacement = sr.actions[0].replacement
        thinking_blocks = [b for b in replacement["message"]["content"] if b["type"] == "thinking"]
        self.assertEqual(len(thinking_blocks), 1)
        self.assertIn("truncated", thinking_blocks[0]["thinking"])
        self.assertNotIn("signature", thinking_blocks[0])


class TestEnvelopeStrip(unittest.TestCase):
    def test_strips_constant_fields(self):
        messages = [
            make_user(0, "hi"),
            make_user(1, "there"),
            make_user(2, "friend"),
        ]
        sr = STRATEGIES["envelope-strip"].func(messages, {})
        # Should strip from messages 1 and 2 (not 0)
        self.assertEqual(sr.messages_replaced, 2)
        for action in sr.actions:
            replacement = action.replacement
            self.assertNotIn("cwd", replacement)
            self.assertNotIn("version", replacement)
            self.assertNotIn("slug", replacement)


class TestToolOutputTrim(unittest.TestCase):
    def test_trims_large_tool_result(self):
        big_content = "line\n" * 200  # 200 lines
        msg = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": big_content},
                ],
            },
        }
        messages = [make_message(0, msg)]
        sr = STRATEGIES["tool-output-trim"].func(messages, {"tool_output_max_lines": 100})
        self.assertEqual(sr.messages_replaced, 1)
        saved = sr.actions[0].original_bytes - sr.actions[0].pruned_bytes
        self.assertGreater(saved, 0)


# ── stale-reads fixtures ─────────────────────────────────────────────────────

_OLD_LINE = "CONFIG_VALUE = 42  # the-line-that-will-be-edited\n"
_OTHER_LINE = "OTHER_VALUE = 99  # edited-region-that-was-never-read\n"
# >500 bytes, contains _OLD_LINE exactly once
_BIG_BODY = ("padding-padding-padding-padding-padding\n" * 10
             + _OLD_LINE
             + "padding-padding-padding-padding-padding\n" * 10)
# >500 bytes, contains _OLD_LINE twice
_BIG_BODY_DUP = ("padding-padding-padding-padding-padding\n" * 10
                 + _OLD_LINE
                 + "padding-padding-padding-padding-padding\n" * 10
                 + _OLD_LINE)
_BIG_WITHOUT_OLD = "padding-padding-padding-padding-padding\n" * 25
_SMALL_BODY = "short file\n" + _OLD_LINE  # well under 500 bytes
_FP = "/proj/src/app.py"


def make_tool_calls(line_idx, calls, cwd="/proj", sidechain=False):
    """Assistant message carrying tool_use blocks. calls = [(id, name, input), ...]."""
    blocks = [
        {"type": "tool_use", "id": cid, "name": name, "input": inp}
        for cid, name, inp in calls
    ]
    msg = {
        "type": "assistant",
        "cwd": cwd,
        "isSidechain": sidechain,
        "message": {"role": "assistant", "content": blocks},
        "uuid": f"uuid-{line_idx}",
    }
    return make_message(line_idx, msg)


def make_tool_results(line_idx, results, cwd="/proj", sidechain=False):
    """User message carrying tool_result blocks. results = [(id, content, is_error), ...]."""
    blocks = []
    for cid, content, is_error in results:
        block = {"type": "tool_result", "tool_use_id": cid, "content": content}
        if is_error:
            block["is_error"] = True
        blocks.append(block)
    msg = {
        "type": "user",
        "cwd": cwd,
        "isSidechain": sidechain,
        "message": {"role": "user", "content": blocks},
        "uuid": f"uuid-{line_idx}",
    }
    return make_message(line_idx, msg)


def _edit_success(line_idx, use_id="e1", cwd="/proj", sidechain=False,
                  text="The file /proj/src/app.py has been updated."):
    return make_tool_results(line_idx, [(use_id, text, False)], cwd=cwd, sidechain=sidechain)


def _result_blocks(action):
    return action.replacement["message"]["content"]


class TestStaleReads(unittest.TestCase):
    def _run(self, messages):
        return STRATEGIES["stale-reads"].func(messages, {})

    def test_successful_covering_edit_trims_read(self):
        messages = [
            make_tool_calls(0, [("r1", "Read", {"file_path": _FP})]),
            make_tool_results(1, [("r1", _BIG_BODY, False)]),
            make_tool_calls(2, [("e1", "Edit", {"file_path": _FP, "old_string": _OLD_LINE,
                                                "new_string": "CONFIG_VALUE = 43\n"})]),
            _edit_success(3),
        ]
        sr = self._run(messages)
        self.assertEqual(sr.messages_replaced, 1)
        self.assertEqual(sr.actions[0].line_index, 1)
        self.assertEqual(_result_blocks(sr.actions[0])[0]["content"],
                         "[stale read - file was later edited, trimmed by cozempic]")

    def test_local_edit_outside_read_window_keeps_read(self):
        """Edit succeeds but replaces a region the (partial) read did not contain."""
        messages = [
            make_tool_calls(0, [("r1", "Read", {"file_path": _FP, "offset": 200, "limit": 100})]),
            make_tool_results(1, [("r1", _BIG_WITHOUT_OLD, False)]),
            make_tool_calls(2, [("e1", "Edit", {"file_path": _FP, "old_string": _OTHER_LINE,
                                                "new_string": "x\n"})]),
            _edit_success(3),
        ]
        sr = self._run(messages)
        self.assertEqual(sr.messages_replaced, 0)

    def test_failed_edit_is_error_keeps_read(self):
        messages = [
            make_tool_calls(0, [("r1", "Read", {"file_path": _FP})]),
            make_tool_results(1, [("r1", _BIG_BODY, False)]),
            make_tool_calls(2, [("e1", "Edit", {"file_path": _FP, "old_string": _OLD_LINE})]),
            make_tool_results(3, [("e1", "No replacement was found.", True)]),
        ]
        self.assertEqual(self._run(messages).messages_replaced, 0)

    def test_failed_edit_textual_signal_keeps_read(self):
        """No is_error flag, but the body says the needle was not found."""
        messages = [
            make_tool_calls(0, [("r1", "Read", {"file_path": _FP})]),
            make_tool_results(1, [("r1", _BIG_BODY, False)]),
            make_tool_calls(2, [("e1", "Edit", {"file_path": _FP, "old_string": _OLD_LINE})]),
            make_tool_results(3, [("e1", "The string to replace was not found in the file.", False)]),
        ]
        self.assertEqual(self._run(messages).messages_replaced, 0)

    def test_pending_edit_without_result_keeps_read(self):
        """Edit call still in flight (no tool_result in transcript) -> outcome unknown."""
        messages = [
            make_tool_calls(0, [("r1", "Read", {"file_path": _FP})]),
            make_tool_results(1, [("r1", _BIG_BODY, False)]),
            make_tool_calls(2, [("e1", "Edit", {"file_path": _FP, "old_string": _OLD_LINE})]),
        ]
        self.assertEqual(self._run(messages).messages_replaced, 0)

    def test_successful_write_trims_read(self):
        messages = [
            make_tool_calls(0, [("r1", "Read", {"file_path": _FP})]),
            make_tool_results(1, [("r1", _BIG_BODY, False)]),
            make_tool_calls(2, [("w1", "Write", {"file_path": _FP, "content": _BIG_BODY})]),
            make_tool_results(3, [("w1", "Successfully wrote to " + _FP, False)]),
        ]
        self.assertEqual(self._run(messages).messages_replaced, 1)

    def test_failed_write_keeps_read(self):
        messages = [
            make_tool_calls(0, [("r1", "Read", {"file_path": _FP})]),
            make_tool_results(1, [("r1", _BIG_BODY, False)]),
            make_tool_calls(2, [("w1", "Write", {"file_path": _FP, "content": "x"})]),
            make_tool_results(3, [("w1", "Permission denied", True)]),
        ]
        self.assertEqual(self._run(messages).messages_replaced, 0)

    def test_failed_read_never_trimmed_even_after_write(self):
        messages = [
            make_tool_calls(0, [("r1", "Read", {"file_path": _FP})]),
            make_tool_results(1, [("r1", "ENOENT: no such file", True)]),
            make_tool_calls(2, [("w1", "Write", {"file_path": _FP, "content": "x"})]),
            make_tool_results(3, [("w1", "Successfully wrote " + _FP, False)]),
        ]
        self.assertEqual(self._run(messages).messages_replaced, 0)

    def test_delayed_results_still_correlated(self):
        """Read result 7 turns later, edit result 2 turns after its call — far
        outside the old pos+1..pos+5 lookup window — must still correlate by
        tool_use_id and trim."""
        messages = [
            make_tool_calls(0, [("r1", "Read", {"file_path": _FP})]),
            make_assistant(1, "thinking about it"),
            make_user(2, "do more"),
            make_assistant(3, "waiting"),
            make_user(4, "still waiting"),
            make_assistant(5, "hmm"),
            make_user(6, "almost"),
            make_tool_results(7, [("r1", _BIG_BODY, False)]),
            make_tool_calls(8, [("e1", "Edit", {"file_path": _FP, "old_string": _OLD_LINE})]),
            make_user(9, "interleaved prompt"),
            _edit_success(10),
        ]
        sr = self._run(messages)
        self.assertEqual(sr.messages_replaced, 1)
        self.assertEqual(sr.actions[0].line_index, 7)

    def test_multiple_reads_same_message_share_one_action(self):
        """Two stale reads in one assistant message, both results in one user
        message: exactly ONE replace action that stubs BOTH blocks."""
        messages = [
            make_tool_calls(0, [
                ("r1", "Read", {"file_path": _FP}),
                ("r2", "Read", {"file_path": _FP, "offset": 1, "limit": 50}),
            ]),
            make_tool_results(1, [("r1", _BIG_BODY, False), ("r2", _BIG_BODY, False)]),
            make_tool_calls(2, [("e1", "Edit", {"file_path": _FP, "old_string": _OLD_LINE})]),
            _edit_success(3),
        ]
        sr = self._run(messages)
        self.assertEqual(sr.messages_replaced, 1)
        self.assertEqual(sr.actions[0].line_index, 1)
        contents = [b["content"] for b in _result_blocks(sr.actions[0])]
        self.assertEqual(contents, [
            "[stale read - file was later edited, trimmed by cozempic]",
            "[stale read - file was later edited, trimmed by cozempic]",
        ])

    def test_multiple_reads_different_files_only_edited_one_trimmed(self):
        other = "/proj/src/other.py"
        messages = [
            make_tool_calls(0, [
                ("r1", "Read", {"file_path": _FP}),
                ("r2", "Read", {"file_path": other}),
            ]),
            make_tool_results(1, [("r1", _BIG_BODY, False), ("r2", _BIG_BODY, False)]),
            make_tool_calls(2, [("e1", "Edit", {"file_path": _FP, "old_string": _OLD_LINE})]),
            _edit_success(3),
        ]
        sr = self._run(messages)
        self.assertEqual(sr.messages_replaced, 1)
        blocks = _result_blocks(sr.actions[0])
        self.assertEqual(blocks[0]["tool_use_id"], "r1")
        self.assertIn("trimmed by cozempic", blocks[0]["content"])
        self.assertEqual(blocks[1]["tool_use_id"], "r2")
        self.assertEqual(blocks[1]["content"], _BIG_BODY)

    def test_different_projects_do_not_cross_invalidate(self):
        rel = "src/app.py"
        messages = [
            make_tool_calls(0, [("r1", "Read", {"file_path": rel})], cwd="/proj/a"),
            make_tool_results(1, [("r1", _BIG_BODY, False)], cwd="/proj/a"),
            make_tool_calls(2, [("e1", "Edit", {"file_path": rel, "old_string": _OLD_LINE})],
                            cwd="/proj/b"),
            _edit_success(3, cwd="/proj/b"),
        ]
        self.assertEqual(self._run(messages).messages_replaced, 0)

        # Sanity: same cwd DOES invalidate
        messages2 = [
            make_tool_calls(0, [("r1", "Read", {"file_path": rel})], cwd="/proj/a"),
            make_tool_results(1, [("r1", _BIG_BODY, False)], cwd="/proj/a"),
            make_tool_calls(2, [("e1", "Edit", {"file_path": rel, "old_string": _OLD_LINE})],
                            cwd="/proj/a"),
            _edit_success(3, cwd="/proj/a"),
        ]
        self.assertEqual(self._run(messages2).messages_replaced, 1)

    def test_sidechain_and_main_chain_isolated(self):
        # Main-chain read, sidechain edit -> keep
        messages = [
            make_tool_calls(0, [("r1", "Read", {"file_path": _FP})], sidechain=False),
            make_tool_results(1, [("r1", _BIG_BODY, False)], sidechain=False),
            make_tool_calls(2, [("e1", "Edit", {"file_path": _FP, "old_string": _OLD_LINE})],
                            sidechain=True),
            _edit_success(3, sidechain=True),
        ]
        self.assertEqual(self._run(messages).messages_replaced, 0)

        # Sidechain read, main-chain edit -> keep
        messages = [
            make_tool_calls(0, [("r1", "Read", {"file_path": _FP})], sidechain=True),
            make_tool_results(1, [("r1", _BIG_BODY, False)], sidechain=True),
            make_tool_calls(2, [("e1", "Edit", {"file_path": _FP, "old_string": _OLD_LINE})],
                            sidechain=False),
            _edit_success(3, sidechain=False),
        ]
        self.assertEqual(self._run(messages).messages_replaced, 0)

        # Sidechain read + sidechain edit -> trim
        messages = [
            make_tool_calls(0, [("r1", "Read", {"file_path": _FP})], sidechain=True),
            make_tool_results(1, [("r1", _BIG_BODY, False)], sidechain=True),
            make_tool_calls(2, [("e1", "Edit", {"file_path": _FP, "old_string": _OLD_LINE})],
                            sidechain=True),
            _edit_success(3, sidechain=True),
        ]
        self.assertEqual(self._run(messages).messages_replaced, 1)

    def test_edit_with_unknown_range_keeps_read(self):
        """Malformed Edit input (no old_string) -> range cannot be established."""
        messages = [
            make_tool_calls(0, [("r1", "Read", {"file_path": _FP})]),
            make_tool_results(1, [("r1", _BIG_BODY, False)]),
            make_tool_calls(2, [("e1", "Edit", {"file_path": _FP, "new_string": "x"})]),
            _edit_success(3),
        ]
        self.assertEqual(self._run(messages).messages_replaced, 0)

    def test_parallel_read_and_edit_same_message_not_later(self):
        messages = [
            make_tool_calls(0, [
                ("r1", "Read", {"file_path": _FP}),
                ("e1", "Edit", {"file_path": _FP, "old_string": _OLD_LINE}),
            ]),
            make_tool_results(1, [
                ("r1", _BIG_BODY, False),
                ("e1", "The file /proj/src/app.py has been updated.", False),
            ]),
        ]
        self.assertEqual(self._run(messages).messages_replaced, 0)

    def test_ambiguous_multimatch_kept_unless_replace_all(self):
        # Two occurrences, plain Edit (must target a unique match) -> ambiguous -> keep
        messages = [
            make_tool_calls(0, [("r1", "Read", {"file_path": _FP})]),
            make_tool_results(1, [("r1", _BIG_BODY_DUP, False)]),
            make_tool_calls(2, [("e1", "Edit", {"file_path": _FP, "old_string": _OLD_LINE})]),
            _edit_success(3),
        ]
        self.assertEqual(self._run(messages).messages_replaced, 0)

        # replace_all replaces every occurrence including the read ones -> trim
        messages[2] = make_tool_calls(2, [("e1", "Edit", {
            "file_path": _FP, "old_string": _OLD_LINE, "replace_all": True})])
        self.assertEqual(self._run(messages).messages_replaced, 1)

    def test_list_content_result_stubbed(self):
        messages = [
            make_tool_calls(0, [("r1", "Read", {"file_path": _FP})]),
            make_tool_results(1, [("r1", [{"type": "text", "text": _BIG_BODY}], False)]),
            make_tool_calls(2, [("e1", "Edit", {"file_path": _FP, "old_string": _OLD_LINE})]),
            _edit_success(3),
        ]
        sr = self._run(messages)
        self.assertEqual(sr.messages_replaced, 1)
        content = _result_blocks(sr.actions[0])[0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["text"],
                         "[stale read - file was later edited, trimmed by cozempic]")

    def test_small_read_result_not_stubbed(self):
        messages = [
            make_tool_calls(0, [("r1", "Read", {"file_path": _FP})]),
            make_tool_results(1, [("r1", _SMALL_BODY, False)]),
            make_tool_calls(2, [("e1", "Edit", {"file_path": _FP, "old_string": _OLD_LINE})]),
            _edit_success(3),
        ]
        self.assertEqual(self._run(messages).messages_replaced, 0)


if __name__ == "__main__":
    unittest.main()
