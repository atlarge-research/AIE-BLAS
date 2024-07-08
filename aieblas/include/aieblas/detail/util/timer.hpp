#pragma once
#include <algorithm>
#include <chrono>
#include <exception>
#include <string>
#include <vector>

template <class Clock>
class Event {
public:
    Event(std::string name, std::chrono::time_point<Clock> time_point)
        : name(name), time_point(time_point) { }

    const std::string name;
    const std::chrono::time_point<Clock> time_point;

    inline bool operator==(const Event<Clock> &rhs) const {
        return name == rhs.name;
    }

    inline bool operator==(const std::string &rhs) const {
        return name == rhs;
    }
};

template <class Clock = std::chrono::high_resolution_clock>
class Timer {
public:
    Timer() { }

    inline const Event<Clock> &time_point(std::string name) {
        return events.emplace_back(name, Clock::now());
    }

    inline const Event<Clock> &time_point() {
        return time_point("event_" + std::to_string(events.size()));
    }

    template <class duration = std::chrono::milliseconds>
    inline duration time(const Event<Clock> &start,
                         const Event<Clock> &end) const {
        return std::chrono::duration_cast<duration>(
            end.time_point - start.time_point);
    }

    template <class duration = std::chrono::milliseconds>
    inline duration time(const Event<Clock> &end) const {
        return time(events.front(), end);
    }

    template <class duration = std::chrono::milliseconds>
    inline duration time() const {
        return time(events.front(), events.back());
    }

    template <class duration = std::chrono::milliseconds>
    inline duration time(const std::string &start,
                         const std::string &end) const {
        return time(get_event(start), get_event(end));
    }

    template <class duration = std::chrono::milliseconds>
    inline duration time(const std::string &end) const {
        return time(events.front(), get_event(end));
    }

private:
    inline const Event<Clock> &get_event(std::string name) const {
        auto it = std::find(events.begin(), events.end(), name);
        if (it == events.end()) {
            throw std::runtime_error("Timer does not contain event \"" + name
                                     + "\"");
        }

        return *it;
    }

    std::vector<Event<Clock>> events;
};
