#pragma once

#include "ninfer/types.h"
#include "models/qwen3_5/frontend/tool_call_grammar.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_5::frontend {

// R10-I4: the next 4-column tab stop. Tabs advance to the stop, never by a byte count:
// column 0/1/2/3 + TAB -> 4, column 4 + TAB -> 8.
[[nodiscard]] constexpr std::size_t next_tab_stop(std::size_t column) noexcept {
    return ((column / 4) + 1) * 4;
}

// R3-06/R10: a line-oriented, streaming-safe fence tracker for the pre-latch content channel.
// It decides which bytes belong to a recognized fenced code block and therefore cannot start a
// top-level tool marker (fence bytes are ordinary content; a tool marker inside a fence never
// latches). Deterministic rules (a practical CommonMark subset, documented so the behavior is
// line-oriented, not full Markdown):
//   * a line starts at the stream start or after a LF and may carry up to 3 spaces of
//     indentation (an opener line; a closing run additionally up to opener_indent + 3);
//   * outside a fence, a run of >= 3 '`' or '~' at line start opens a fence with that
//     character; extra fence characters on the opener line extend the run length;
//   * inside a fence, a run of the same character, indented up to opener_indent + 3
//     spaces, of length >= the opener length, followed only by format whitespace up to
//     the line end (CR, spaces, tabs) closes the fence; a different character, a shorter
//     run, or a non-whitespace byte after a close run keeps the line as fence content
//     (no nested fences);
//   * a backtick info string containing a backtick cancels the opener (the line is
//     inline code); CRLF framing of the close line keeps the close valid;
//   * an unclosed fence stays open through EOF (suppression is the safe direction).
class ToolCallFenceTracker {
public:
    enum class Verdict : std::uint8_t {
        Pass,    // the byte is not fence structure: the marker machine handles it
        Content, // the byte is fence structure: publish as ordinary content, no candidate
    };
    [[nodiscard]] Verdict consume(char byte) noexcept;
    // True while a fence is still open (the pre-latch stream ended in an unclosed fence).
    [[nodiscard]] bool open() const noexcept { return in_fence_; }

private:
    enum class Phase : std::uint8_t {
        LineIndent, // line start: counting indentation before a run or body byte
        LineRun,    // a fence-character run (opener candidate or close candidate)
        OpenerTail, // the rest of an opener line (run extension or info string)
        LineTail,   // format whitespace after a close run: the close is still valid
        LineBody,   // the line is classified (fence content or ordinary text)
    };
    bool in_fence_ = false;
    char fence_char_ = '\0';
    std::size_t fence_len_ = 0;
    std::size_t fence_indent_ = 0;
    Phase phase_ = Phase::LineIndent;
    char run_char_ = '\0';
    std::size_t run_len_ = 0;
    std::size_t indent_ = 0;
    bool close_ok_ = false;
    bool opener_line_ = false;
};

// R3-06/R10: the shadow marker scan — the same classify_tool_marker_prefix transition over
// suppressed bytes (fence content or indented literal content), never latching: it counts
// the complete top-level markers the suppression removed from the entry machine. One
// instance per suppression class so the counters never double-count (a byte is fed to
// exactly one shadow: fence bytes to the fence shadow, indented-literal bytes to the
// indented shadow, R10-I7).
class ToolCallShadowMarkerScan {
public:
    explicit ToolCallShadowMarkerScan(ToolCallSyntaxMode syntax) : syntax_(syntax) {}
    void consume(char byte);
    [[nodiscard]] std::uint32_t complete() const noexcept { return complete_; }

private:
    ToolCallSyntaxMode syntax_;
    std::string candidate_;
    std::uint32_t complete_ = 0;
};

// R10 (Round 10, R10-I1): the single pre-trigger entry classifier shared by the production
// pre-latch parser and the CPU grammar-constraint core. It owns every pre-latch-sensitive
// state — fence suppression, indented-literal classification, the marker candidate, the
// held formatting whitespace, the intent gate, and the deterministic failed-candidate
// rescan — so the two consumers are governed by one authoritative machine and can never
// drift on entry eligibility.
//
// It decides only whether a tool region starts. Once a marker is accepted
// (FeedResult::triggered), the caller owns the region bytes and this scanner is no longer
// consulted for them (R10-I11: literal suppression applies only before entry latch; inside
// an owned region the wire grammar owns the bytes).
//
// Indented-literal rule (R10-I2/I3/I5): before a tool entry has latched, a possible tool
// marker beginning a physical line at visual indentation column 4 or greater is classified
// as literal content and cannot become executable tool markup. This is a deterministic
// tool-safety rule, not complete CommonMark parsing.
class ToolCallEntryScanner {
public:
    ToolCallEntryScanner(ToolCallSyntaxMode syntax, ToolCallIntentPolicy intent);

    struct FeedResult {
        // The bytes determined to be ordinary content by this feed, in publication order.
        std::string visible;
        // A complete marker was accepted as the region entry.
        bool triggered = false;
        // Bytes consumed from this feed (all of it when not triggered; through the last
        // byte of the accepted candidate when triggered). The caller seeds the region with
        // region_prefix and appends the unconsumed feed remainder.
        std::size_t consumed = 0;
        // The full prefix that seeds the region parser: the held formatting whitespace (if
        // any) followed by the accepted marker, plus the re-fed suffixes of the failed
        // candidates the rescan walked (each enclosing frame appends the bytes after its
        // trigger byte, mirroring the deterministic failed-candidate rescan contract).
        std::string region_prefix;
    };

    [[nodiscard]] FeedResult feed(std::string_view bytes);

    // The intent gate locked: a visible (non-formatting-whitespace) content byte was
    // committed before a trigger under RequireToolAtContentStart. While locked no later
    // marker may trigger — a complete marker is ordinary content. Always false under
    // TemplateCompatible.
    [[nodiscard]] bool locked() const noexcept { return entry_locked_; }
    // The bytes held back after the published content: the formatting whitespace prefix
    // plus the pending marker candidate (the candidate is always a strict marker prefix,
    // or empty).
    [[nodiscard]] std::string held_tail() const { return held_ws_ + candidate_; }
    // The pending marker candidate (empty when none is held).
    [[nodiscard]] std::string_view pending_candidate() const noexcept { return candidate_; }
    // R3-14: deterministic pre-latch rescan counter (bytes re-fed after a NotMarker).
    [[nodiscard]] std::uint64_t rescan_steps() const noexcept { return rescan_steps_; }
    // R3-06/R10: complete top-level markers suppressed by recognized fenced code.
    [[nodiscard]] std::uint32_t fenced_markers_suppressed() const noexcept {
        return fence_shadow_.complete();
    }
    // R10-03: complete top-level markers suppressed because their '<' began on an indented
    // literal line (visual column >= 4 outside a fence).
    [[nodiscard]] std::uint32_t indented_markers_suppressed() const noexcept {
        return indent_shadow_.complete();
    }
    // True while the pre-latch stream ended inside a recognized unclosed fence.
    [[nodiscard]] bool ended_in_unclosed_fence() const noexcept { return fence_.open(); }

private:
    // One marker-machine byte; true on trigger. R3-05: a NotMarker result publishes the
    // failed candidate head (trailing format whitespace held) and re-feeds the bytes from
    // the next '<' through this same transition (the deterministic rescan; R3-14 counts
    // the re-fed bytes). Recursion depth is bounded by the '<' count of one candidate,
    // which is bounded by kMaxToolHeaderBytes. The re-fed bytes already passed the fence
    // and column layers (they were counted when first fed), so the rescan recurses within
    // the marker layer only.
    [[nodiscard]] bool marker_byte(char byte, std::string& visible, std::string& region_prefix);
    // The single intent-lock funnel (R6-05): visible (non-formatting-whitespace) content
    // locks the gate under RequireToolAtContentStart. Formatting whitespace never locks;
    // form feed is not format whitespace here, so it locks like any other visible byte.
    void lock_if_visible(std::string_view bytes) noexcept;

    ToolCallSyntaxMode syntax_;
    ToolCallIntentPolicy intent_;
    ToolCallFenceTracker fence_;
    ToolCallShadowMarkerScan fence_shadow_;
    ToolCallShadowMarkerScan indent_shadow_;
    // Visual indentation column of the current physical line (R10 §5.2): counts only the
    // line's leading indentation — spaces +1, tabs to the next 4-column stop, CR no
    // column — and is frozen once the line's first non-formatting byte arrives. LF resets
    // the physical line.
    std::size_t visual_column_ = 0;
    // True from the stream start / an LF until the line's first non-formatting byte: the
    // indentation-scanning window in which spaces and tabs advance visual_column_.
    bool at_line_start_ = true;
    // Set when the line's first non-formatting byte arrives at visual column >= 4 outside
    // a fence: every byte of the line through the LF is literal content.
    bool literal_line_ = false;
    std::string held_ws_;    // format whitespace since the last published byte (held: may precede a marker)
    std::string candidate_;  // held bytes that may become a top-level marker
    std::uint64_t rescan_steps_ = 0;
    // R6-05 intent gate: a visible content byte was committed before a latch; under
    // RequireToolAtContentStart no later marker may latch.
    bool entry_locked_ = false;
};

} // namespace ninfer::models::qwen3_5::frontend
