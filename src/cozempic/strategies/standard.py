"""Standard-tier strategies: recommended pruning with cross-message correlation."""

from __future__ import annotations

import hashlib
import json
import os
import re

from ..helpers import get_content_blocks, get_msg_type, hashable_str, is_protected, msg_bytes, set_content_blocks, text_of
from ..registry import strategy
from ..types import Message, PruneAction, StrategyResult
from ._config import coerce_choice, coerce_non_negative_int, coerce_ordered_pair

_THINKING_MODES: tuple[str, ...] = ("remove", "truncate", "signature-only")


@strategy("thinking-blocks", "Truncate or remove thinking/signature blocks", "standard", "2-5%")
def strategy_thinking_blocks(messages: list[Message], config: dict) -> StrategyResult:
    """Remove or truncate thinking blocks and signatures from assistant messages.

    Modes (via config['thinking_mode']):
        'remove'         - Remove thinking blocks entirely (default)
        'truncate'       - Keep first 200 chars of thinking
        'signature-only' - Only strip signature fields
    """
    mode = coerce_choice(config, "thinking_mode", _THINKING_MODES, default="remove")
    actions: list[PruneAction] = []
    total_orig = sum(b for _, _, b in messages)
    total_pruned = 0
    replaced = 0

    for pos, (idx, msg, size) in enumerate(messages):
        if is_protected(msg):
            continue
        if get_msg_type(msg) != "assistant":
            continue

        blocks = get_content_blocks(msg)
        if not blocks:
            continue

        new_blocks = []
        changed = False
        for block in blocks:
            btype = block.get("type", "")
            if btype == "thinking":
                changed = True
                if mode == "remove":
                    continue
                elif mode == "truncate":
                    thinking = block.get("thinking", "")
                    new_block = {k: v for k, v in block.items() if k != "signature"}
                    if len(thinking) > 200:
                        new_block["thinking"] = thinking[:200] + "...[truncated]"
                    new_blocks.append(new_block)
                elif mode == "signature-only":
                    new_block = {k: v for k, v in block.items() if k != "signature"}
                    new_blocks.append(new_block)
                    changed = new_block != block
            else:
                if "signature" in block:
                    changed = True
                    new_blocks.append({k: v for k, v in block.items() if k != "signature"})
                else:
                    new_blocks.append(block)

        if changed:
            new_msg = set_content_blocks(msg, new_blocks)
            new_size = msg_bytes(new_msg)
            saved = size - new_size
            if saved > 0:
                actions.append(PruneAction(
                    line_index=idx,
                    action="replace",
                    reason=f"thinking-blocks ({mode})",
                    original_bytes=size,
                    pruned_bytes=new_size,
                    replacement=new_msg,
                ))
                total_pruned += saved
                replaced += 1

    return StrategyResult(
        strategy_name="thinking-blocks",
        actions=actions,
        original_bytes=total_orig,
        pruned_bytes=total_pruned,
        messages_affected=replaced,
        messages_removed=0,
        messages_replaced=replaced,
        summary=f"Processed {replaced} thinking blocks (mode={mode})",
    )


@strategy("tool-output-trim", "Trim large tool_result blocks (>8KB or >100 lines)", "standard", "1-8%")
def strategy_tool_output_trim(messages: list[Message], config: dict) -> StrategyResult:
    """Trim oversized tool results while preserving structure."""
    max_bytes = coerce_non_negative_int(config, "tool_output_max_bytes", default=8192)
    max_lines = coerce_non_negative_int(config, "tool_output_max_lines", default=100)

    actions: list[PruneAction] = []
    total_orig = sum(b for _, _, b in messages)
    total_pruned = 0
    replaced = 0

    # T2.3: Collect tool IDs already summarized by microcompact — don't trim those
    compacted_tool_ids: set[str] = set()
    for _, msg, _ in messages:
        if msg.get("type") == "system" and msg.get("subtype") == "microcompact_boundary":
            _ctids = msg.get("compactedToolIds", [])
            for tid in (_ctids if isinstance(_ctids, list) else []):
                if isinstance(tid, str):  # unhashable/non-str element -> skip (R6 crash class)
                    compacted_tool_ids.add(tid)

    for pos, (idx, msg, size) in enumerate(messages):
        if is_protected(msg):
            continue
        blocks = get_content_blocks(msg)
        if not blocks:
            continue

        new_blocks = []
        changed = False
        for block in blocks:
            if not isinstance(block, dict):  # non-dict element preserved, never .get()'d (R6)
                new_blocks.append(block)
                continue
            if block.get("type") == "tool_result":
                # Skip tool results already microcompacted
                tool_use_id = hashable_str(block.get("tool_use_id"))
                if tool_use_id and tool_use_id in compacted_tool_ids:
                    new_blocks.append(block)
                    continue
                content = block.get("content", "")
                if isinstance(content, str):
                    content_bytes = len(content.encode("utf-8"))
                    content_lines = content.count("\n") + 1
                    if content_bytes > max_bytes or content_lines > max_lines:
                        lines = content.split("\n")
                        if len(lines) > max_lines:
                            keep = max_lines // 2
                            trimmed = (
                                lines[:keep]
                                + [f"\n... [{len(lines) - max_lines} lines trimmed by cozempic] ...\n"]
                                + lines[-keep:]
                            )
                            new_content = "\n".join(trimmed)
                        else:
                            half = max_bytes // 2
                            new_content = (
                                content[:half]
                                + f"\n... [{content_bytes - max_bytes} bytes trimmed by cozempic] ...\n"
                                + content[-half:]
                            )
                        new_blocks.append({**block, "content": new_content})
                        changed = True
                        continue
                elif isinstance(content, list):
                    block_json = json.dumps(content, separators=(",", ":"))
                    if len(block_json.encode("utf-8")) > max_bytes:
                        trimmed_content = []
                        for sub in content:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                text = sub.get("text", "")
                                if isinstance(text, str) and len(text.encode("utf-8", "surrogateescape")) > max_bytes:
                                    half = max_bytes // 2
                                    sub = {**sub, "text": text[:half] + "\n...[trimmed by cozempic]...\n" + text[-half:]}
                            trimmed_content.append(sub)
                        new_blocks.append({**block, "content": trimmed_content})
                        changed = True
                        continue
            new_blocks.append(block)

        if changed:
            new_msg = set_content_blocks(msg, new_blocks)
            new_size = msg_bytes(new_msg)
            saved = size - new_size
            if saved > 0:
                actions.append(PruneAction(
                    line_index=idx,
                    action="replace",
                    reason="tool-output-trim",
                    original_bytes=size,
                    pruned_bytes=new_size,
                    replacement=new_msg,
                ))
                total_pruned += saved
                replaced += 1

    return StrategyResult(
        strategy_name="tool-output-trim",
        actions=actions,
        original_bytes=total_orig,
        pruned_bytes=total_pruned,
        messages_affected=replaced,
        messages_removed=0,
        messages_replaced=replaced,
        summary=f"Trimmed {replaced} oversized tool outputs",
    )


_STALE_READ_TOOL_NAMES = ("Read", "read")
_STALE_WRITE_TOOL_NAMES = ("Write", "write")
_STALE_EDIT_TOOL_NAMES = ("Edit", "edit", "MultiEdit")
_STALE_READ_STUB = "[stale read - file was later edited, trimmed by cozempic]"
_STALE_MIN_CONTENT_BYTES = 500

# Secondary failure signal for harnesses that return errors WITHOUT is_error.
# Kept to SPECIFIC phrases used by file-edit tools — a successful edit result
# ("The file has been updated") never matches. The primary signal is is_error.
_STALE_FAILURE_RE = re.compile(
    r"(was not found in the file|"
    r"no replacements? (?:was|were)? ?found|"
    r"found \d+ (?:matches?|matching sections)\b|"
    r"did not match|"
    r"could not(?: be)? (?:find|apply|edit|write|update|save)|"
    r"failed to (?:apply|edit|write|update|save)|"
    r"permission denied|"
    r"no such file)",
    re.IGNORECASE,
)


def _stale_call_scope(msg: dict) -> tuple[str, bool]:
    """Identity scope of a tool call: project (normalized cwd) + chain kind
    (main chain vs sidechain/subagent). Reads and writes only invalidate each
    other INSIDE the same scope — same relative path in another project,
    worktree, or sidechain is a different file state."""
    cwd = msg.get("cwd") if isinstance(msg, dict) else None
    cwd = os.path.normpath(cwd) if isinstance(cwd, str) and cwd else ""
    sidechain = bool(msg.get("isSidechain")) if isinstance(msg, dict) else False
    return (cwd, sidechain)


def _stale_norm_path(path) -> str:
    return os.path.normpath(path) if isinstance(path, str) and path else ""


def _stale_result_text(block: dict) -> str:
    """All text carried by a tool_result block (str content or list of text parts)."""
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            sub["text"] for sub in content
            if isinstance(sub, dict) and isinstance(sub.get("text"), str)
        )
    return ""


def _stale_result_failed(block: dict) -> bool:
    """True only with positive evidence the tool call FAILED. An absent result
    (call still in flight / delayed beyond the transcript) is NOT a failure —
    callers treat missing results separately as outcome-unknown."""
    if block.get("is_error"):
        return True
    return bool(_STALE_FAILURE_RE.search(_stale_result_text(block)))


def _stale_content_bytes(block: dict) -> int:
    content = block.get("content", "")
    if isinstance(content, str):
        return len(content.encode("utf-8", "surrogateescape"))
    try:
        return len(json.dumps(content, separators=(",", ":")).encode("utf-8", "surrogateescape"))
    except (TypeError, ValueError):
        return 0


def _stale_stub_block(block: dict) -> dict:
    """Replace a stale read result's body with the stub, preserving structure
    (list content keeps non-text parts, e.g. images, untouched)."""
    content = block.get("content", "")
    if isinstance(content, list):
        new_content = []
        did_stub = False
        for sub in content:
            if isinstance(sub, dict) and sub.get("type") == "text" and isinstance(sub.get("text"), str):
                if not did_stub:
                    new_content.append({**sub, "text": _STALE_READ_STUB})
                    did_stub = True
                # further text parts are the same read body — collapse them
            else:
                new_content.append(sub)
        if not did_stub:
            new_content = [{"type": "text", "text": _STALE_READ_STUB}]
        return {**block, "content": new_content}
    return {**block, "content": _STALE_READ_STUB}


@strategy("stale-reads", "Remove file reads superseded by later edits", "standard", "0.5-2%")
def strategy_stale_reads(messages: list[Message], config: dict) -> StrategyResult:
    """Trim a file-read result only when a LATER SUCCESSFUL write demonstrably
    covers the bytes that were read.

    Invalidation rules (fail-safe: when in doubt, keep the read):
      * identity  — read/write are correlated within (cwd, isSidechain) scope +
                    normalized path; other projects/worktrees/sidechains are
                    independent;
      * success   — the write's tool_result must exist (delayed results are
                    found by tool_use_id at ANY distance) and not be an error;
                    failed/in-flight writes never invalidate;
      * coverage  — Write covers the whole file; Edit/MultiEdit only invalidate
                    when an exact `old_string` is found inside THAT read's own
                    result text (unique match, or replace_all). A local edit in
                    another region, or an edit whose range cannot be
                    established, keeps the read;
      * ordering  — the write call must come strictly AFTER the read call
                    (parallel calls in one message never invalidate each other).

    Trimming is aggregated per result message, so multiple stale reads whose
    results share one message produce a SINGLE replace action and never
    clobber each other (the executor keeps only one replacement per line).
    """
    actions: list[PruneAction] = []
    total_orig = sum(b for _, _, b in messages)
    total_pruned = 0

    # tool_use_id -> (pos, line_index, result block). Built across ALL messages
    # with no distance window: results can lag far behind their calls
    # (background tasks / interleaved tool batches). First occurrence wins.
    results_by_id: dict[str, tuple[int, int, dict]] = {}

    reads: list[dict] = []
    # (scope, path) -> list of successful-or-pending write records
    writes: dict[tuple, list[dict]] = {}

    for pos, (idx, msg, _size) in enumerate(messages):
        if is_protected(msg):
            continue
        scope = _stale_call_scope(msg)
        for block in get_content_blocks(msg):
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_result":
                use_id = hashable_str(block.get("tool_use_id"))
                if use_id and use_id not in results_by_id:
                    results_by_id[use_id] = (pos, idx, block)
                continue
            if btype != "tool_use":
                continue

            use_id = hashable_str(block.get("id"))
            if not use_id:
                continue
            tool_input = block.get("input")
            if not isinstance(tool_input, dict):
                continue
            name = block.get("name", "")
            path = _stale_norm_path(tool_input.get("file_path"))
            if not path:
                continue

            if name in _STALE_READ_TOOL_NAMES:
                reads.append({"pos": pos, "use_id": use_id, "scope": scope, "path": path})
            elif name in _STALE_WRITE_TOOL_NAMES:
                writes.setdefault((scope, path), []).append(
                    {"pos": pos, "use_id": use_id, "full_cover": True, "old_strings": None}
                )
            elif name in _STALE_EDIT_TOOL_NAMES:
                old_strings = _stale_edit_old_strings(name, tool_input)
                writes.setdefault((scope, path), []).append({
                    "pos": pos,
                    "use_id": use_id,
                    "full_cover": False,
                    "old_strings": old_strings,
                    "replace_all": tool_input.get("replace_all") is True,
                })

    # Decide staleness PER READ CALL (not per message — a message may carry
    # several reads of different files with different fates).
    stale_read_ids: set[str] = set()
    for read in reads:
        read_result = results_by_id.get(read["use_id"])
        # Read failed / result missing / result unreadable: there is no valid
        # read body whose coverage could be established — keep it.
        if read_result is None or _stale_result_failed(read_result[2]):
            continue
        read_text = _stale_result_text(read_result[2])
        if not read_text:
            continue

        for write in writes.get((read["scope"], read["path"]), ()):
            if write["pos"] <= read["pos"]:
                continue  # same message (parallel) or earlier — not a later write
            write_result = results_by_id.get(write["use_id"])
            if write_result is None:
                continue  # in flight / delayed beyond transcript — outcome unknown
            if _stale_result_failed(write_result[2]):
                continue  # failed write never invalidates
            if write["full_cover"]:
                stale_read_ids.add(read["use_id"])
                break
            if _stale_edit_covers_read(write["old_strings"], bool(write.get("replace_all")), read_text):
                stale_read_ids.add(read["use_id"])
                break

    # Aggregate stubs per RESULT message: one PruneAction per carrier line.
    plan: dict[int, set[str]] = {}
    for use_id in stale_read_ids:
        rpos, ridx, rblock = results_by_id[use_id]
        if is_protected(messages[rpos][1]):
            continue
        if _stale_content_bytes(rblock) <= _STALE_MIN_CONTENT_BYTES:
            continue
        plan.setdefault(ridx, set()).add(use_id)

    idx_to_pos = {idx: pos for pos, (idx, _m, _s) in enumerate(messages)}
    for fidx, ids in plan.items():
        fpos = idx_to_pos.get(fidx)
        if fpos is None:
            continue
        _fidx, fmsg, fsize = messages[fpos]
        new_blocks = []
        for block in get_content_blocks(fmsg):
            if (isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and hashable_str(block.get("tool_use_id")) in ids):
                new_blocks.append(_stale_stub_block(block))
            else:
                new_blocks.append(block)
        new_msg = set_content_blocks(fmsg, new_blocks)
        new_size = msg_bytes(new_msg)
        saved = fsize - new_size
        if saved > 0:
            actions.append(PruneAction(
                line_index=fidx,
                action="replace",
                reason="stale-read (file later edited)",
                original_bytes=fsize,
                pruned_bytes=new_size,
                replacement=new_msg,
            ))
            total_pruned += saved

    replaced = len(actions)
    return StrategyResult(
        strategy_name="stale-reads",
        actions=actions,
        original_bytes=total_orig,
        pruned_bytes=total_pruned,
        messages_affected=replaced,
        messages_removed=0,
        messages_replaced=replaced,
        summary=f"Trimmed {replaced} stale file read results",
    )


def _stale_edit_old_strings(name: str, tool_input: dict) -> list[str] | None:
    """Extract the exact strings an edit replaced, in file coordinates.

    Returns None when the covered range cannot be established (malformed
    input / empty needles) — such an edit can NEVER prove coverage, so reads
    are kept. MultiEdit yields one needle per sub-edit."""
    if name == "MultiEdit":
        edits = tool_input.get("edits")
        if not isinstance(edits, list) or not edits:
            return None
        needles: list[str] = []
        for sub in edits:
            if not isinstance(sub, dict):
                return None
            old = sub.get("old_string")
            if not isinstance(old, str) or not old:
                return None
            needles.append(old)
        return needles or None
    old = tool_input.get("old_string")
    if not isinstance(old, str) or not old:
        return None
    return [old]


def _stale_edit_covers_read(old_strings: list[str] | None, replace_all: bool,
                            read_text: str) -> bool:
    """True only if a successful edit demonstrably changed bytes INSIDE the
    read's own result text:

      * an `old_string` absent from the read body edited a different region
        (local/partial edit) — no coverage;
      * exactly one occurrence in the read body — certain coverage (a
        non-replace_all Edit that SUCCEEDED was globally unique);
      * multiple occurrences only count with replace_all (every occurrence,
        including the read one, was replaced); ambiguity otherwise -> keep.
    """
    if not old_strings:
        return False  # range unknown
    for old in old_strings:
        hits = read_text.count(old)
        if hits == 1 or (hits > 1 and replace_all):
            return True
    return False


@strategy("system-reminder-dedup", "Deduplicate repeated <system-reminder> tags", "standard", "0.1-3%")
def strategy_system_reminder_dedup(messages: list[Message], config: dict) -> StrategyResult:
    """Remove duplicate system-reminder content, keeping only the first occurrence."""
    actions: list[PruneAction] = []
    total_orig = sum(b for _, _, b in messages)
    total_pruned = 0
    replaced = 0

    reminder_pattern = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
    seen_hashes: set[str] = set()

    for pos, (idx, msg, size) in enumerate(messages):
        if is_protected(msg):
            continue
        blocks = get_content_blocks(msg)
        if not blocks:
            continue

        new_blocks = []
        changed = False
        for block in blocks:
            if block.get("type") in ("text", "tool_result"):
                text = block.get("text", "") or (block.get("content", "") if isinstance(block.get("content"), str) else "")
                if not text:
                    new_blocks.append(block)
                    continue

                reminders = reminder_pattern.findall(text)
                if reminders:
                    new_text = text
                    for reminder in reminders:
                        h = hashlib.md5(reminder.encode()).hexdigest()
                        if h in seen_hashes:
                            new_text = new_text.replace(reminder, "")
                            changed = True
                        else:
                            seen_hashes.add(h)

                    if changed:
                        new_text = re.sub(r"\n{3,}", "\n\n", new_text).strip()
                        if block.get("type") == "text":
                            new_blocks.append({**block, "text": new_text})
                        elif block.get("type") == "tool_result" and isinstance(block.get("content"), str):
                            new_blocks.append({**block, "content": new_text})
                        else:
                            new_blocks.append(block)
                    else:
                        new_blocks.append(block)
                else:
                    new_blocks.append(block)
            else:
                new_blocks.append(block)

        if changed:
            new_msg = set_content_blocks(msg, new_blocks)
            new_size = msg_bytes(new_msg)
            saved = size - new_size
            if saved > 0:
                actions.append(PruneAction(
                    line_index=idx,
                    action="replace",
                    reason="system-reminder-dedup",
                    original_bytes=size,
                    pruned_bytes=new_size,
                    replacement=new_msg,
                ))
                total_pruned += saved
                replaced += 1

    return StrategyResult(
        strategy_name="system-reminder-dedup",
        actions=actions,
        original_bytes=total_orig,
        pruned_bytes=total_pruned,
        messages_affected=replaced,
        messages_removed=0,
        messages_replaced=replaced,
        summary=f"Deduped system-reminders in {replaced} messages ({len(seen_hashes)} unique)",
    )


@strategy("tool-result-age", "Compact old tool results by age — minify mid-age, stub old", "standard", "10-40%")
def strategy_tool_result_age(messages: list[Message], config: dict) -> StrategyResult:
    """Three-tier age-based tool result compaction.

    Tool results decay in value exponentially. A file read from 50 turns ago
    is stale (the file has been edited since) and wastes tokens. Claude can
    always re-read if needed.

    Tiers (configurable via config):
      Recent (0 to mid_age turns):  untouched
      Mid-age (mid_age to old_age): minify JSON, strip diff context lines
      Old (old_age+):               replace content with compact stub

    Research: JetBrains validated observation masking on SWE-bench — matched
    LLM summarization quality at zero compute cost.
    """
    mid_age, old_age = coerce_ordered_pair(
        config, "tool_result_mid_age", "tool_result_old_age", defaults=(15, 40)
    )

    actions: list[PruneAction] = []
    total_orig = sum(b for _, _, b in messages)
    total_pruned = 0
    replaced = 0

    # Count actual user prompts (not tool_result wrappers which also have type="user")
    def _is_user_prompt(msg: dict) -> bool:
        if get_msg_type(msg) != "user":
            return False
        content = msg.get("message", {}).get("content", "")
        # tool_result messages have list content with tool_result blocks
        if isinstance(content, list):
            return not any(b.get("type") == "tool_result" for b in content if isinstance(b, dict))
        return isinstance(content, str)

    total_turns = sum(1 for _, msg, _ in messages if _is_user_prompt(msg))

    # Build turn index: for each message position, how many user prompts precede it?
    turn_count = 0
    msg_turn: list[int] = []
    for _, msg, _ in messages:
        if _is_user_prompt(msg):
            turn_count += 1
        msg_turn.append(turn_count)

    for pos, (idx, msg, size) in enumerate(messages):
        if is_protected(msg):
            continue

        blocks = get_content_blocks(msg)
        if not blocks:
            continue

        has_tool_result = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in blocks)
        if not has_tool_result:
            continue

        turns_ago = total_turns - msg_turn[pos]

        if turns_ago < mid_age:
            continue  # Recent — keep verbatim

        new_blocks = []
        changed = False

        for block in blocks:
            if block.get("type") != "tool_result":
                new_blocks.append(block)
                continue

            content = block.get("content", "")
            if not isinstance(content, str) or len(content) < 100:
                new_blocks.append(block)
                continue

            tool_use_id = block.get("tool_use_id", "")

            if turns_ago >= old_age:
                # OLD: replace with compact stub
                stub = _build_stub(block, blocks, messages, pos)
                new_blocks.append({**block, "content": stub})
                changed = True
            else:
                # MID-AGE: minify content
                compacted = _minify_tool_content(content)
                if compacted != content:
                    new_blocks.append({**block, "content": compacted})
                    changed = True
                else:
                    new_blocks.append(block)

        if changed:
            new_msg = set_content_blocks(msg, new_blocks)
            new_size = msg_bytes(new_msg)
            saved = size - new_size
            if saved > 0:
                actions.append(PruneAction(
                    line_index=idx,
                    action="replace",
                    reason=f"tool-result-age ({turns_ago} turns ago)",
                    original_bytes=size,
                    pruned_bytes=new_size,
                    replacement=new_msg,
                ))
                total_pruned += saved
                replaced += 1

    return StrategyResult(
        strategy_name="tool-result-age",
        actions=actions,
        original_bytes=total_orig,
        pruned_bytes=total_pruned,
        messages_affected=replaced,
        messages_removed=0,
        messages_replaced=replaced,
        summary=f"Compacted {replaced} old tool results ({total_pruned / 1024:.0f}KB saved)",
    )


def _build_stub(block: dict, all_blocks: list[dict], messages: list[Message], pos: int) -> str:
    """Build a compact stub for an old tool result."""
    content = block.get("content", "")
    tool_use_id = block.get("tool_use_id", "")
    content_len = len(content)
    line_count = content.count("\n") + 1 if content else 0

    # Try to find the matching tool_use to get tool name and input
    tool_name = ""
    tool_path = ""
    for search_pos in range(max(0, pos - 10), pos + 1):
        if search_pos >= len(messages):
            break
        _, search_msg, _ = messages[search_pos]
        for b in get_content_blocks(search_msg):
            if b.get("type") == "tool_use" and b.get("id") == tool_use_id:
                tool_name = b.get("name", "")
                tool_input = b.get("input", {})
                tool_path = (
                    tool_input.get("file_path", "")
                    or tool_input.get("path", "")
                    or tool_input.get("pattern", "")
                    or tool_input.get("command", "")[:80]
                )
                break

    parts = ["[cozempic"]
    if tool_name:
        parts.append(f": {tool_name}")
    if tool_path:
        parts.append(f" {tool_path}")
    parts.append(f" — {line_count} lines, {content_len / 1024:.1f}KB]")

    return "".join(parts)


def _minify_tool_content(content: str) -> str:
    """Minify mid-age tool result content: strip JSON whitespace, collapse diff context."""
    # Try JSON minification first
    try:
        parsed = json.loads(content)
        minified = json.dumps(parsed, separators=(",", ":"))
        if len(minified) < len(content) * 0.85:  # Only if meaningful savings
            return minified
    except (json.JSONDecodeError, TypeError):
        pass

    # Collapse diff context lines — but ONLY for a GENUINE unified diff. The gate
    # requires the unified-diff ENVELOPE (a `--- `/`+++ ` file-header pair or a
    # `diff ` command line) AND a real hunk header. A lone coincidental `@@ … @@`
    # line in non-diff output (a git-log fragment, CI text, an indented config
    # block) no longer triggers collapse — and even past the gate,
    # _collapse_diff_context only collapses context lines INSIDE a hunk, so it can
    # never wholesale-destroy indented non-diff content (the audit P1).
    if "\0" not in content and _looks_like_unified_diff(content):
        collapsed = _collapse_diff_context(content)
        if collapsed != content:
            return collapsed

    return content


# A real unified-diff hunk header, anchored at line start: "@@ -12,7 +12,9 @@".
_UNIFIED_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.M)
# The file-header pair every unified diff carries: a "--- …" line immediately
# followed by a "+++ …" line.
_DIFF_FILE_HDR_RE = re.compile(r"^--- .*\n\+\+\+ ", re.M)


def _looks_like_unified_diff(content: str) -> bool:
    """True only for content with the unified-diff envelope (so we never collapse
    non-diff output that merely contains a hunk-shaped line)."""
    if not _UNIFIED_HUNK_RE.search(content):
        return False
    if _DIFF_FILE_HDR_RE.search(content):
        return True
    return content.startswith("diff ") or "\ndiff --git " in content or "\ndiff " in content


def _collapse_diff_context(diff_text: str) -> str:
    """Strip unchanged context lines from unified diffs, keep +/- and headers.

    Context lines (" "-prefixed) are collapsed ONLY while inside a hunk (after an
    `@@` header). Lines before the first hunk, and any non-hunk content, are kept
    verbatim — so a stray indented block cannot be destroyed even if the gate
    passed on a real diff that also contains trailing prose."""
    lines = diff_text.split("\n")
    result = []
    context_run = 0
    in_hunk = False

    def _flush():
        nonlocal context_run
        if context_run > 0:
            result.append(f"  [...{context_run} unchanged lines...]")
            context_run = 0

    for line in lines:
        if line.startswith("@@"):
            _flush()
            in_hunk = True
            result.append(line)
        elif line.startswith(("diff ", "---", "+++", "+", "-")):
            _flush()
            result.append(line)
        elif in_hunk and line.startswith(" "):
            context_run += 1
        else:
            # A non-context line: we're no longer inside a hunk's body. Reset
            # in_hunk so indented content AFTER the hunk (e.g. a git-log-p second
            # commit's message body, or trailing prose) is kept verbatim and never
            # collapsed (the audit P1 — in_hunk was set but never reset).
            in_hunk = False
            _flush()
            result.append(line)

    _flush()
    collapsed = "\n".join(result)
    return collapsed if len(collapsed) < len(diff_text) else diff_text
