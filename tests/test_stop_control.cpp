#include "serve/stop_control.h"

#include <chrono>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer::serve;
using Clock = StopControl::Clock;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool contains(std::string_view text, std::string_view needle) {
    return text.find(needle) != std::string_view::npos;
}

// Records what the control asked the console and the process to do.
struct Observed {
    std::vector<StopConsoleLine> lines;
    std::vector<OperationalRecord> records;
    int stops = 0;
    int exits = 0;

    StopControlActions actions() {
        return {.show     = [this](const StopConsoleLine& line) { lines.push_back(line); },
                .record   = [this](const OperationalRecord& record) { records.push_back(record); },
                .exit_now = [this] { ++exits; }};
    }

    [[nodiscard]] std::string line() const { return lines.empty() ? "" : lines.back().text; }
};

const Clock::time_point t0 = Clock::time_point{} + std::chrono::hours(1);
constexpr Clock::duration kWindow = StopControl::kConfirmWindow;
constexpr Clock::duration kTick   = std::chrono::milliseconds(1);

} // namespace

int main() {
    int failures = 0;

    {
        // Before serving, events keep their default action (a Ctrl+C during loading ends it).
        Observed seen;
        StopControl control(true, seen.actions());
        failures += check(!control.handle(StopEvent::Interrupt, t0) &&
                              !control.handle(StopEvent::Terminate, t0) && seen.lines.empty(),
                          "an inactive control must leave events to their default action");
    }

    {
        // With a prefix cache file: one Ctrl+C only asks; a second inside the window stops.
        Observed seen;
        StopControl control(true, seen.actions());
        control.serve([&] { ++seen.stops; });
        failures += check(control.handle(StopEvent::Interrupt, t0) && seen.stops == 0,
                          "a single Ctrl+C must not stop the server");
        failures += check(seen.lines.size() == 1 && seen.lines.back().prompt &&
                              seen.line() == "Press Ctrl+C again within 5 s to save the prefix "
                                             "cache and close",
                          "the first Ctrl+C must prompt to save the prefix cache and close");
        failures += check(control.expire(t0 + kWindow) == t0 + kWindow && seen.lines.size() == 1,
                          "the prompt must stay until its window has passed");
        failures += check(!control.expire(t0 + kWindow + kTick) && seen.line().empty(),
                          "an unanswered prompt must be withdrawn after its window");

        // After the window a Ctrl+C asks again; the second press on the window's edge still counts.
        const Clock::time_point t1 = t0 + kWindow + kTick;
        control.handle(StopEvent::Interrupt, t1);
        failures += check(seen.stops == 0 && seen.lines.back().prompt,
                          "a Ctrl+C after the window must prompt again, not stop");
        control.handle(StopEvent::Interrupt, t1 + kWindow);
        failures += check(seen.stops == 1, "a confirming Ctrl+C must stop the server");
        failures += check(!seen.records.empty() &&
                              seen.records.back().severity == OperationalSeverity::Info &&
                              contains(seen.records.back().message, "Ctrl+C: stopping") &&
                              contains(seen.records.back().message, "requests are cancelled"),
                          "the stop must be logged as a Ctrl+C stop that cancels requests");
        failures += check(!seen.lines.back().prompt &&
                              contains(seen.line(), "saving the prefix cache") &&
                              contains(seen.line(), "Ctrl+C twice exits without saving"),
                          "the bottom line must show the save in progress while stopping");

        // While stopping, abandoning the save needs its own confirmed pair of presses.
        const Clock::time_point t2 = t1 + kWindow + kTick;
        control.handle(StopEvent::Interrupt, t2);
        failures += check(seen.exits == 0 && seen.lines.back().prompt &&
                              contains(seen.line(), "exit without saving the prefix cache"),
                          "one Ctrl+C while stopping must only prompt to exit without saving");
        control.expire(t2 + kWindow + kTick);
        failures += check(contains(seen.line(), "saving the prefix cache") &&
                              !seen.lines.back().prompt,
                          "a withdrawn prompt while stopping must restore the save status");
        control.handle(StopEvent::Interrupt, t2 + 2 * kWindow);
        control.handle(StopEvent::Interrupt, t2 + 2 * kWindow + kTick);
        failures += check(seen.exits == 1 && seen.stops == 1,
                          "a confirmed Ctrl+C while stopping must exit at once");
        failures += check(seen.records.back().severity == OperationalSeverity::Warning &&
                              contains(seen.records.back().message, "previous file is kept"),
                          "exiting before the save must say the previous file is kept");
    }

    {
        // Ctrl+Break, console close and SIGTERM stop at once and never exit early.
        Observed seen;
        StopControl control(true, seen.actions());
        control.serve([&] { ++seen.stops; });
        control.handle(StopEvent::Interrupt, t0);
        failures += check(control.handle(StopEvent::Terminate, t0 + kTick) && seen.stops == 1 &&
                              !contains(seen.records.back().message, "Ctrl+C:"),
                          "a terminate event must stop at once, even with a Ctrl+C pending");
        control.handle(StopEvent::Terminate, t0 + 2 * kTick);
        failures += check(seen.stops == 1 && seen.exits == 0,
                          "a repeated terminate event must neither stop again nor exit");
        // The Ctrl+C pressed before the stop must not confirm one pressed during it.
        control.handle(StopEvent::Interrupt, t0 + 3 * kTick);
        failures += check(seen.exits == 0 && seen.lines.back().prompt,
                          "a Ctrl+C from before the stop must not confirm an exit");
    }

    {
        // Without a prefix cache file the texts do not mention one.
        Observed seen;
        StopControl control(false, seen.actions());
        control.serve([&] { ++seen.stops; });
        control.handle(StopEvent::Interrupt, t0);
        failures += check(seen.line() == "Press Ctrl+C again within 5 s to close",
                          "without a cache file the prompt must offer to close");
        control.handle(StopEvent::Interrupt, t0 + kTick);
        failures += check(seen.stops == 1 && !contains(seen.records.back().message, "prefix") &&
                              !contains(seen.line(), "prefix"),
                          "without a cache file the stop must not mention the prefix cache");
    }

    {
        // After listen() returns, no event reaches the server; finish() restores default actions.
        Observed seen;
        StopControl control(true, seen.actions());
        control.serve([&] { ++seen.stops; });
        control.end_serving();
        control.handle(StopEvent::Interrupt, t0);
        control.handle(StopEvent::Interrupt, t0 + kTick);
        failures += check(seen.stops == 0 && seen.exits == 1,
                          "after listen() returns, confirmed Ctrl+C must exit, not stop");
        control.finish();
        failures += check(!control.handle(StopEvent::Interrupt, t0 + 2 * kTick) &&
                              seen.line().empty(),
                          "a finished control must clear the line and decline events");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
