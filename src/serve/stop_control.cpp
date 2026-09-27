#include "serve/stop_control.h"

#include <string>
#include <utility>

namespace ninfer::serve {

namespace {

const std::string kWithin =
    "Press Ctrl+C again within " + std::to_string(StopControl::kConfirmWindow.count()) + " s to ";

} // namespace

StopControl::StopControl(bool saves_prefix_cache, StopControlActions actions)
    : saves_prefix_cache_(saves_prefix_cache), actions_(std::move(actions)) {}

void StopControl::serve(std::function<void()> stop_server) {
    std::lock_guard lock(mutex_);
    stop_server_ = std::move(stop_server);
    phase_       = Phase::Serving;
    confirm_until_.reset();
}

void StopControl::end_serving() {
    std::lock_guard lock(mutex_);
    stop_server_ = nullptr;
    if (phase_ == Phase::Serving) { phase_ = Phase::Stopping; }
}

void StopControl::finish() {
    std::lock_guard lock(mutex_);
    phase_       = Phase::Inactive;
    stop_server_ = nullptr;
    confirm_until_.reset();
    actions_.show({});
}

bool StopControl::handle(StopEvent event, Clock::time_point now) {
    std::lock_guard lock(mutex_);
    if (phase_ == Phase::Inactive) { return false; }
    if (event == StopEvent::Terminate) {
        if (phase_ == Phase::Serving) { begin_stop_locked(false); }
        return true;
    }
    if (!confirm_until_ || now > *confirm_until_) {
        confirm_until_ = now + kConfirmWindow;
        actions_.show(prompt_line_locked());
        return true;
    }
    confirm_until_.reset();
    if (phase_ == Phase::Serving) {
        begin_stop_locked(true);
        return true;
    }
    actions_.record({.severity = OperationalSeverity::Warning,
                     .message  = std::string("Ctrl+C: exiting before the stop finished") +
                                (saves_prefix_cache_ ? " | an unfinished prefix cache save is "
                                                       "abandoned; the previous file is kept"
                                                     : "")});
    actions_.exit_now();
    return true;
}

std::optional<StopControl::Clock::time_point> StopControl::expire(Clock::time_point now) {
    std::lock_guard lock(mutex_);
    if (confirm_until_ && now > *confirm_until_) {
        confirm_until_.reset();
        actions_.show(resting_line_locked());
    }
    return confirm_until_;
}

void StopControl::begin_stop_locked(bool interrupt) {
    phase_ = Phase::Stopping;
    confirm_until_.reset();
    actions_.record({.severity = OperationalSeverity::Info,
                     .message  = std::string(interrupt ? "Ctrl+C: " : "") +
                                "stopping | running and queued requests are cancelled" +
                                (saves_prefix_cache_
                                     ? " | Ctrl+C twice exits without saving the prefix cache"
                                     : "")});
    actions_.show(resting_line_locked());
    // Only closes a socket and signals the Engine's worker; it does not wait.
    if (stop_server_) { stop_server_(); }
}

StopConsoleLine StopControl::prompt_line_locked() const {
    if (phase_ == Phase::Serving) {
        return {.text   = kWithin + (saves_prefix_cache_ ? "save the prefix cache and close"
                                                         : "close"),
                .prompt = true};
    }
    return {.text   = kWithin + (saves_prefix_cache_ ? "exit without saving the prefix cache"
                                                     : "exit without waiting"),
            .prompt = true};
}

StopConsoleLine StopControl::resting_line_locked() const {
    if (phase_ != Phase::Stopping) { return {}; }
    return {.text = saves_prefix_cache_
                        ? "Closing: saving the prefix cache | Ctrl+C twice exits without saving"
                        : "Closing | Ctrl+C twice exits at once"};
}

} // namespace ninfer::serve
