#include "runtime/engine/request_record.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::runtime;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// RequestRecord is not movable (atomic members); construct it in place with a contract that
// exercises only the template aliases the record actually uses.
struct FakeModelContract {
    struct PreparedPrompt {};
    struct OutputSession {};
    struct RequestBasePlan {};
    struct NgramArchive { struct Request {}; };
    using SequenceHandle = std::uint64_t;
};

using Record = RequestRecord<FakeModelContract>;

ToolCallPreviewSnapshot make_snapshot(const char* name, const char* partial) {
    ToolCallPreviewSnapshot snapshot;
    snapshot.calls.push_back(
        ToolCallPreviewSnapshot::Call{.name = name, .partial_arguments = partial});
    return snapshot;
}

struct RecordingSink : OutputSink {
    std::vector<std::string> published;
    std::vector<ToolCallPreviewSnapshot> previews;
    int timing_calls = 0;

    void start(GenerationStart) override {}
    void progress(PromptProgress) override {}
    void timing(GenerationTimingObservation) override { ++timing_calls; }
    void publish(OutputDelta delta) override { published.push_back(std::move(delta.text)); }
    void publish_tool_call_preview(ToolCallPreviewSnapshot snapshot) override {
        previews.push_back(std::move(snapshot));
    }
};

// N1: a mixed batch reaches the sink in event order: delta first, preview second.
int test_dispatch_mixed_batch_order() {
    Record record{1, 1, FakeModelContract::PreparedPrompt{},
                  FakeModelContract::OutputSession{},
                  PromptSummary{}, 0.0, ResolvedRequestOptions{},
                  OutputConsumerMode::Streaming, GenerationObservationOptions{},
                  Record::Clock::time_point{}, Record::Clock::time_point{}};
    record.events.push_back(OutputDelta{.channel = OutputChannel::Content, .text = "hello"});
    record.events.push_back(make_snapshot("write", "{\"path\":"));
    RecordingSink sink;
    const std::exception_ptr error = dispatch_stream_events(record.events, &sink);
    int failures = 0;
    failures += check(error == nullptr, "N1: a well-behaved sink must not report an error");
    failures += check(sink.published.size() == 1 && sink.published[0] == "hello",
                      "N1: the delta must reach the sink unchanged");
    failures += check(sink.previews.size() == 1 && sink.previews[0].calls.size() == 1 &&
                          sink.previews[0].calls[0].name == "write" &&
                          sink.previews[0].calls[0].partial_arguments == "{\"path\":",
                      "N1: the preview snapshot must reach the sink in batch order");
    return failures;
}

// N2: a delta-only batch must not synthesize preview events.
int test_dispatch_delta_only_batch() {
    Record record{2, 2, FakeModelContract::PreparedPrompt{},
                  FakeModelContract::OutputSession{},
                  PromptSummary{}, 0.0, ResolvedRequestOptions{},
                  OutputConsumerMode::Streaming, GenerationObservationOptions{},
                  Record::Clock::time_point{}, Record::Clock::time_point{}};
    for (const char* text : {"a", "b"}) {
        record.events.push_back(OutputDelta{.channel = OutputChannel::Content, .text = text});
    }
    RecordingSink sink;
    const std::exception_ptr error = dispatch_stream_events(record.events, &sink);
    int failures = 0;
    failures += check(error == nullptr, "N2: a well-behaved sink must not report an error");
    failures += check(sink.published.size() == 2 && sink.previews.empty(),
                      "N2: a delta-only batch must not synthesize preview events");
    return failures;
}

// N3: the default publish_tool_call_preview is a no-op, and a null sink consumes the batch
// without any calls.
int test_dispatch_default_and_null_sink() {
    struct MinimalSink : OutputSink {
        std::vector<std::string> published;
        void start(GenerationStart) override {}
        void progress(PromptProgress) override {}
        void timing(GenerationTimingObservation) override {}
        void publish(OutputDelta delta) override { published.push_back(std::move(delta.text)); }
    };
    Record record{3, 3, FakeModelContract::PreparedPrompt{},
                  FakeModelContract::OutputSession{},
                  PromptSummary{}, 0.0, ResolvedRequestOptions{},
                  OutputConsumerMode::Streaming, GenerationObservationOptions{},
                  Record::Clock::time_point{}, Record::Clock::time_point{}};
    record.events.push_back(make_snapshot("write", "{\"path\":"));
    MinimalSink minimal;
    const std::exception_ptr minimal_error = dispatch_stream_events(record.events, &minimal);
    int failures = 0;
    failures += check(minimal_error == nullptr && minimal.published.empty(),
                      "N3: the default preview handler must be inert");
    record.events.clear();
    record.events.push_back(OutputDelta{.channel = OutputChannel::Content, .text = "x"});
    record.events.push_back(make_snapshot("write", "{\"path\":\"/a\""));
    const std::exception_ptr null_error = dispatch_stream_events(record.events, nullptr);
    failures += check(null_error == nullptr,
                      "N3: a null sink must consume the batch without any sink calls");
    return failures;
}

// N4: a sink failure stops further delivery after the failing event and is reported as
// exception_ptr.
int test_dispatch_sink_failure_stops_delivery() {
    struct FailingSink : OutputSink {
        int publish_calls = 0;
        int preview_calls = 0;
        void start(GenerationStart) override {}
        void progress(PromptProgress) override {}
        void timing(GenerationTimingObservation) override {}
        void publish(OutputDelta) override {
            ++publish_calls;
            if (publish_calls == 2) { throw std::runtime_error("second delta"); }
        }
        void publish_tool_call_preview(ToolCallPreviewSnapshot) override { ++preview_calls; }
    };
    Record record{4, 4, FakeModelContract::PreparedPrompt{},
                  FakeModelContract::OutputSession{},
                  PromptSummary{}, 0.0, ResolvedRequestOptions{},
                  OutputConsumerMode::Streaming, GenerationObservationOptions{},
                  Record::Clock::time_point{}, Record::Clock::time_point{}};
    record.events.push_back(OutputDelta{.channel = OutputChannel::Content, .text = "one"});
    record.events.push_back(OutputDelta{.channel = OutputChannel::Content, .text = "two"});
    record.events.push_back(make_snapshot("write", "{\"path\":"));
    FailingSink sink;
    std::exception_ptr error;
    try {
        error = dispatch_stream_events(record.events, &sink);
    } catch (...) {
        error = std::current_exception();
    }
    int failures = 0;
    failures += check(error != nullptr, "N4: the sink failure must be reported as exception_ptr");
    failures += check(sink.publish_calls == 2 && sink.preview_calls == 0,
                      "N4: delivery must stop after the failing event");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_dispatch_mixed_batch_order();
    failures += test_dispatch_delta_only_batch();
    failures += test_dispatch_default_and_null_sink();
    failures += test_dispatch_sink_failure_stops_delivery();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
