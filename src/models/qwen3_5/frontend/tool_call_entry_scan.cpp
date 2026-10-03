#include "models/qwen3_5/frontend/tool_call_entry_scan.h"
#include <algorithm>


namespace ninfer::models::qwen3_5::frontend {

ToolCallFenceTracker::Verdict ToolCallFenceTracker::consume(char byte) noexcept {
    if (byte == '\n') {
        // R3-06: a close line is a run of the opener character of at least the opener length,
        // followed only by format whitespace (CR, spaces, tabs) up to the line end — CRLF and
        // LF-only framing both keep the close valid. The opener line never closes the fence.
        if (in_fence_ && !opener_line_ && close_ok_ &&
            (phase_ == Phase::LineRun || phase_ == Phase::LineTail) && run_len_ >= fence_len_) {
            in_fence_  = false;
            fence_len_ = 0;
        }
        opener_line_ = false;
        close_ok_    = false;
        run_len_     = 0;
        indent_      = 0;
        phase_       = Phase::LineIndent;
        return in_fence_ ? Verdict::Content : Verdict::Pass;
    }
    if (!in_fence_) {
        // Outside a fence: only a run of >= 3 '`' or '~' at line start (indent <= 3) is fence
        // structure; everything else is left to the marker machine.
        if (phase_ == Phase::LineIndent) {
            if (byte == ' ') {
                ++indent_;
                if (indent_ > 3) { phase_ = Phase::LineBody; }
                return Verdict::Pass;
            }
            if (byte == '`' || byte == '~') {
                phase_    = Phase::LineRun;
                run_char_ = byte;
                run_len_  = 1;
                return Verdict::Content;
            }
            phase_ = Phase::LineBody;
            return Verdict::Pass;
        }
        if (phase_ == Phase::LineRun) {
            if (byte == run_char_) {
                ++run_len_;
                if (run_len_ == 3) {
                    // The run opens the fence at its third character; the rest of the line is
                    // the info string, and further run characters extend the fence length.
                    in_fence_     = true;
                    fence_char_   = byte;
                    fence_len_    = 3;
                    fence_indent_ = indent_;
                    opener_line_  = true;
                }
                return Verdict::Content;
            }
            phase_ = Phase::LineBody;
            return Verdict::Pass;
        }
        return Verdict::Pass; // LineBody
    }
    // Inside a fence.
    if (opener_line_) {
        if (phase_ == Phase::LineRun) {
            if (byte == fence_char_) { ++fence_len_; return Verdict::Content; }
            // The run ended: the info string starts (the cancel rule only applies from the
            // info string on, where a backtick means inline code, not a longer run).
            phase_ = Phase::OpenerTail;
            return Verdict::Content;
        }
        if (fence_char_ == '`' && byte == '`') {
            // A backtick in a backtick opener's info string: the line is inline code, not a
            // fence (CommonMark). Cancel the opener; the line is ordinary content from here.
            in_fence_    = false;
            fence_len_   = 0;
            opener_line_ = false;
            phase_       = Phase::LineBody;
            return Verdict::Pass;
        }
        return Verdict::Content; // the info string
    }
    // Other lines inside a fence (always content; no nested fences).
    if (phase_ == Phase::LineIndent) {
        if (byte == ' ') {
            ++indent_;
            if (indent_ > fence_indent_ + 3) { phase_ = Phase::LineBody; }
            return Verdict::Content;
        }
        if (byte == fence_char_) {
            phase_    = Phase::LineRun;
            run_char_ = byte;
            run_len_  = 1;
            close_ok_ = true;
            return Verdict::Content;
        }
        phase_ = Phase::LineBody;
        return Verdict::Content;
    }
    if (phase_ == Phase::LineRun) {
        if (byte == run_char_) { ++run_len_; return Verdict::Content; }
        if (is_tool_format_whitespace(byte)) { phase_ = Phase::LineTail; }
        else { close_ok_ = false; phase_ = Phase::LineBody; }
        return Verdict::Content;
    }
    if (phase_ == Phase::LineTail) {
        if (!is_tool_format_whitespace(byte)) { close_ok_ = false; phase_ = Phase::LineBody; }
        return Verdict::Content;
    }
    return Verdict::Content; // LineBody
}

void ToolCallShadowMarkerScan::consume(char byte) {
    if (candidate_.empty()) {
        if (byte == '<') { candidate_.push_back(byte); }
        return;
    }
    candidate_.push_back(byte);
    ToolOpenTag marker = {};
    const ToolMarkerStatus state = classify_tool_marker_prefix(candidate_, marker, syntax_);
    if (state == ToolMarkerStatus::Complete) {
        ++complete_;
        candidate_.clear();
    } else if (state == ToolMarkerStatus::NotMarker) {
        const std::size_t next = candidate_.find('<', 1);
        candidate_ = (next == std::string::npos) ? std::string{} : candidate_.substr(next);
    }
}

ToolCallEntryScanner::ToolCallEntryScanner(ToolCallSyntaxMode syntax, ToolCallIntentPolicy intent)
    : syntax_(syntax), intent_(intent), fence_shadow_(syntax), indent_shadow_(syntax) {}

void ToolCallEntryScanner::lock_if_visible(std::string_view bytes) noexcept {
    if (intent_ != ToolCallIntentPolicy::RequireToolAtContentStart || entry_locked_) { return; }
    for (const char byte : bytes) {
        if (!is_tool_format_whitespace(byte)) {
            entry_locked_ = true;
            return;
        }
    }
}

bool ToolCallEntryScanner::marker_byte(char byte, std::string& visible, std::string& region_prefix) {
    if (!candidate_.empty()) {
        candidate_.push_back(byte);
        ToolOpenTag marker = {};
        const ToolMarkerStatus state = classify_tool_marker_prefix(candidate_, marker, syntax_);
        if (state == ToolMarkerStatus::Complete) {
            if (intent_ == ToolCallIntentPolicy::RequireToolAtContentStart && entry_locked_) {
                // R6-05: visible content was committed before this marker: the policy
                // locks the turn to text. The complete marker is ordinary content and can
                // never latch. A failed marker-like prefix cannot bypass the gate: the
                // NotMarker path publishes the candidate head through the same funnel,
                // which already locked the gate.
                visible += held_ws_;
                held_ws_.clear();
                visible += candidate_;
                candidate_.clear();
                return false;
            }
            region_prefix = std::move(held_ws_);
            held_ws_.clear();
            region_prefix += candidate_;
            candidate_.clear();
            return true;
        }
        if (state == ToolMarkerStatus::NotMarker) {
            // R3-05 (generalized F8): publish the failed candidate's head up to the next '<'
            // (the head's trailing format whitespace stays held, R3-07, so streaming output
            // equals one-shot output), and re-feed the remaining bytes through this same
            // transition. They already passed the fence and column layers, so the rescan
            // recurses within the marker layer only.
            std::string failed = std::move(candidate_);
            candidate_.clear();
            visible += held_ws_;
            held_ws_.clear();
            const std::size_t next = failed_marker_candidate_rescan_start(failed);
            const std::string_view head =
                std::string_view(failed).substr(0, next == std::string_view::npos ? failed.size() : next);
            std::size_t keep = head.size();
            while (keep > 0 && is_tool_format_whitespace(head[keep - 1])) { --keep; }
            visible += head.substr(0, keep);
            lock_if_visible(head.substr(0, keep));
            if (keep < head.size()) { held_ws_ = std::string(head.substr(keep)); }
            if (next != std::string_view::npos) {
                const std::string_view rest = std::string_view(failed).substr(next);
                rescan_steps_ += rest.size(); // R3-14: deterministic work counter
                for (std::size_t j = 0; j < rest.size(); ++j) {
                    if (marker_byte(rest[j], visible, region_prefix)) {
                        // The trigger happened while re-feeding this slice: the region is the
                        // latch base (held formatting whitespace plus the accepted marker,
                        // already in region_prefix) plus this slice's bytes after the trigger
                        // byte; enclosing frames append their own suffixes up the recursion.
                        region_prefix.append(rest, j + 1, std::string_view::npos);
                        return true;
                    }
                }
            }
        }
        return false;
    }
    if (byte == '<') {
        candidate_.push_back(byte);
    } else if (is_tool_format_whitespace(byte)) {
        held_ws_.push_back(byte);
    } else {
        visible += held_ws_;
        held_ws_.clear();
        visible.push_back(byte);
        lock_if_visible(std::string_view(&byte, 1));
    }
    return false;
}

ToolCallEntryScanner::FeedResult ToolCallEntryScanner::feed(std::string_view bytes) {
    FeedResult out;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        const char byte = bytes[i];
        const ToolCallFenceTracker::Verdict fence_verdict = fence_.consume(byte);
        // Column bookkeeping (R10 §5.2/I3/I4): the visual column counts only the line's
        // leading indentation (spaces +1, tabs to the next 4-column stop, CR no column)
        // until the line's first non-formatting byte arrives, and is frozen for the rest of
        // the line; LF resets the physical line. Runs on every pre-latch byte (fence bytes
        // included) so the line state stays deterministic across chunk partitions; the
        // literal decision below only fires on non-fence bytes (R10-I7: fence
        // classification has precedence).
        if (byte == '\n') {
            // The LF that terminates a literal line is fed to the indented shadow so a
            // partial shadow candidate resets (a marker never spans a line end), then the
            // line state resets on the LF.
            if (literal_line_) { indent_shadow_.consume(byte); }
            visual_column_  = 0;
            literal_line_   = false;
            at_line_start_  = true;
        } else if ((byte == ' ' || byte == '\t') && at_line_start_) {
            if (byte == ' ') {
                ++visual_column_;
            } else {
                visual_column_ = next_tab_stop(visual_column_);
            }
        } else if (!is_tool_format_whitespace(byte) && at_line_start_) {
            // The line's first meaningful byte (CR is format whitespace and never
            // advances a column, so it cannot end the indentation window): at visual
            // indentation column >= 4 outside a fence the whole line is a literal
            // (R10-I2/I3: columns 0..3 stay eligible, column >= 4 is literal).
            at_line_start_ = false;
            if (fence_verdict != ToolCallFenceTracker::Verdict::Content &&
                visual_column_ >= 4) {
                literal_line_ = true;
            }
        }
        if (fence_verdict == ToolCallFenceTracker::Verdict::Content) {
            // R2-I6 (CR6): a fence byte is ordinary content and cannot continue a top-level
            // marker. The suppressed marker count sees it (fence shadow, R10-I7: fence has
            // precedence over the indented shadow).
            fence_shadow_.consume(byte);
            if (is_tool_format_whitespace(byte)) {
                // N-08: a fence Content byte that is format whitespace is held instead of
                // published, so the streamed output equals the one-shot entry (which rtrims
                // the whitespace before an accepted region). The fence tracker's verdicts
                // are unchanged; only the publication timing of whitespace changes.
                if (!candidate_.empty()) {
                    out.visible += held_ws_;
                    held_ws_.clear();
                    out.visible += candidate_;
                    candidate_.clear();
                }
                held_ws_.push_back(byte); // held like any other whitespace: may precede a latch
                continue;
            }
            out.visible += held_ws_;
            held_ws_.clear();
            out.visible += candidate_;
            candidate_.clear();
            out.visible.push_back(byte);
            lock_if_visible(std::string_view(&byte, 1));
            continue;
        }
        if (literal_line_) {
            // R10-01: an indented literal line (first meaningful byte at visual column
            // >= 4): every byte of the line through the LF is ordinary content — the held
            // indentation is published (not trimmed, R10 §5.4), no marker may latch, and
            // the indented shadow counts the suppressed markers.
            out.visible += held_ws_;
            held_ws_.clear();
            if (!candidate_.empty()) {
                // Cannot happen (a candidate never straddles a line start: a newline inside
                // a candidate is a NotMarker break), but the funnel stays safe.
                out.visible += candidate_;
                candidate_.clear();
            }
            out.visible.push_back(byte);
            lock_if_visible(std::string_view(&byte, 1));
            indent_shadow_.consume(byte);
            continue;
        }
        if (marker_byte(byte, out.visible, out.region_prefix)) {
            out.triggered = true;
            out.consumed  = i + 1;
            return out;
        }
    }
    return out;
}

ToolCallLineIndentationScan::ToolCallLineIndentationScan(std::string_view pre_latch,
                                                         std::string_view region)
    : pre_latch_size_(pre_latch.size()) {
    const std::size_t P    = pre_latch.size();
    const std::size_t total = P + region.size();
    auto byte_at = [P, pre_latch, region](std::size_t i) -> char {
        return i < P ? pre_latch[i] : region[i - P];
    };

    std::size_t line_start      = 0;
    std::size_t column          = 0;
    std::size_t first_nonspace  = 0;
    std::size_t column_at_first = 0;
    bool        seen_nonspace   = false;

    for (std::size_t i = 0; i <= total; ++i) {
        const bool line_end = (i == total) || byte_at(i) == '\n';
        if (line_end) {
            // A line with no non-formatting byte: first_nonspace sits at the line end
            // (outside the line's byte range), so no marker inside the line can match it.
            lines_.push_back(Line{line_start, seen_nonspace ? first_nonspace : i,
                                  seen_nonspace ? column_at_first : 0});
            line_start      = i + 1;
            column          = 0;
            seen_nonspace   = false;
            if (i == total) { break; }
            continue;
        }
        if (seen_nonspace) { continue; }
        const char b = byte_at(i);
        if (b == ' ') {
            column += 1;
        } else if (b == '\t') {
            column = next_tab_stop(column);
        } else if (b != '\r') {
            // The line's first non-formatting byte; the accumulated column is its visual
            // indentation (the Round-10 rule).
            seen_nonspace   = true;
            first_nonspace  = i;
            column_at_first = column;
        }
        // '\r' changes no column and is not a non-formatting byte for this rule.
    }
}

bool ToolCallLineIndentationScan::marker_is_indented_literal(std::size_t marker_at) const noexcept {
    if (lines_.empty()) { return false; }
    const std::size_t full_m = pre_latch_size_ + marker_at;
    // The line containing full_m is the last line with start <= full_m (lines are sorted by
    // start and tile the text contiguously).
    const auto it = std::upper_bound(
        lines_.begin(), lines_.end(), full_m,
        [](std::size_t value, const Line& line) { return value < line.start; });
    if (it == lines_.begin()) { return false; }
    const Line& line = *std::prev(it);
    return full_m == line.first_nonspace && line.column_at_first >= 4;
}

} // namespace ninfer::models::qwen3_5::frontend
