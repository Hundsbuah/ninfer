// F10: the reserved tool-calls-only constrained-decoding mode fails engine option
// validation with an explicit error before any device work. Off is the accepted default
// and leaves the option validation (and therefore the sampling configuration) untouched.
#include "runtime/engine/model_instance.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ninfer::EngineOptions base_options() {
    ninfer::EngineOptions options;
    options.artifact_path = "artifact-for-option-validation.ninfer";
    return options;
}

} // namespace

int main() {
    int failures = 0;

    // Off is the default and passes validation; the normal sampling configuration is
    // untouched by validation (validation rejects nothing for off).
    const auto off = base_options();
    failures += check(off.constrained_tool_decoding == ninfer::ConstrainedToolDecoding::Off,
                      "the constrained tool decoding default is not off");
    bool off_rejected = false;
    try {
        ninfer::runtime::validate_engine_options(off);
    } catch (const std::exception&) {
        off_rejected = true;
    }
    failures += check(!off_rejected, "off constrained tool decoding was rejected at startup");

    // The reserved mode fails with the explicit expected error.
    auto reserved = base_options();
    reserved.constrained_tool_decoding = ninfer::ConstrainedToolDecoding::ToolCallsOnly;
    bool invalid_argument = false;
    std::string message;
    try {
        ninfer::runtime::validate_engine_options(reserved);
    } catch (const std::invalid_argument& error) {
        invalid_argument = true;
        message = error.what();
    } catch (const std::exception&) {
    }
    failures += check(
        invalid_argument &&
            message == "--constrained-tool-decoding=tool-calls-only is not implemented in "
                           "this build; use off",
        "tool-calls-only constrained decoding did not fail with the expected error");

    // The rejection does not disturb the normal option combination: the same options with
    // off still validate.
    auto after = base_options();
    bool after_rejected = false;
    try {
        ninfer::runtime::validate_engine_options(after);
    } catch (const std::exception&) {
        after_rejected = true;
    }
    failures += check(!after_rejected, "off validation changed after the reserved-mode rejection");
    return failures == 0 ? 0 : 1;
}
