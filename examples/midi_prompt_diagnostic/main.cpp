// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// mrt2_midi_prompt_diagnostic
//
// Offline diagnostic renderer that drives the official C++ MLXEngine MIDI
// path. It parses a Standard MIDI File, schedules note_on/note_off events via
// MLXEngine::set_note_on/off, blends cached prompt embeddings through
// reblend_musiccoca_tokens(), modulates CFG/temperature/top-k, and writes a
// float WAV plus a JSON report.

#include <magentart/mlx_engine.h>

#include "../common/cpp/magenta_paths.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

using namespace magentart::core;

namespace {

constexpr int kSampleRate = 48000;
constexpr int kFps = 25;
constexpr double kFrameSeconds = 1.0 / kFps;
constexpr double kDefaultBpm = 120.0;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct MidiNote {
    int start_tick = 0;
    int end_tick = 0;
    int pitch = 0;
    int velocity = 0;
    int channel = 0;
    double start_seconds = 0.0;
    double end_seconds = 0.0;
};

struct MidiEvent {
    double time_seconds = 0.0;
    int pitch = 0;
    bool on = false;
};

struct MidiData {
    int format = 0;
    int tracks = 0;
    int ppq = 96;
    int end_tick = 0;
    double bpm = kDefaultBpm;
    double duration_seconds = 0.0;
    std::string track_name;
    std::string time_signature = "4/4";
    std::vector<MidiNote> notes;
    std::vector<MidiEvent> events;
};

struct Slot {
    std::string id;
    std::string label;
    std::string prompt;
    std::string audio_recipe;
    float temperature = 1.0f;
    int top_k = 80;
    float cfg_musiccoca = 5.0f;
    float cfg_notes = 5.0f;
    float cfg_drums = 0.0f;
};

struct SegmentMetrics {
    float peak = 0.0f;
    float rms = 0.0f;
};

const std::array<Slot, 4> kSlots = {{
    {
        "solo_piano_chords",
        "Solo Piano Chords",
        "solo acoustic grand piano only, dry close microphone, sustained minor jazz chord voicings, no drums, no bass, no synths",
        "percussive piano-like chord attacks from the provided MIDI progression",
        0.72f,
        28,
        6.8f,
        6.4f,
        0.0f,
    },
    {
        "chiptune_square_arps",
        "Chiptune Square Arps",
        "8-bit chiptune square wave lead, bright retro video game synth, rapid staccato arpeggios, sharp digital beeps, no drums",
        "bright square-wave 16th-note arpeggio from each MIDI chord",
        1.34f,
        192,
        7.0f,
        4.0f,
        0.0f,
    },
    {
        "distorted_808_drums",
        "Distorted 808 Drums",
        "heavy distorted 808 sub bass with punchy trap drums, loud kick and snare, aggressive electronic club rhythm",
        "808 kick/sub rhythm, snare, and hat pattern with MIDI roots",
        1.18f,
        156,
        6.4f,
        3.2f,
        7.0f,
    },
    {
        "choir_string_drone",
        "Choir String Drone",
        "wide cinematic string orchestra and choir pad, slow ambient drone, huge reverb, soft tape noise, no percussion",
        "long slow string and choir-like sustained pad from each MIDI chord",
        0.62f,
        44,
        6.6f,
        6.2f,
        0.0f,
    },
}};

struct RenderConfig {
    std::filesystem::path midi_path = "assets/Am-Minor Prog 01 (i-VI-v-iv).mid";
    std::filesystem::path output_path =
        "outputs/polyrhythm_prompt_modulation/magenta_cpp_midi_control_matrix_8s.wav";
    std::filesystem::path report_path =
        "outputs/polyrhythm_prompt_modulation/magenta_cpp_midi_control_matrix_8s.report.json";
    std::filesystem::path audio_prompt_dir =
        "outputs/polyrhythm_prompt_modulation/cpp_audio_prompt_guides";
    std::string model_name = "mrt2_small";
    std::string resource_dir = magentart::paths::get_resources_dir();
    std::string model_subfolder = "musiccoca";
    bool use_audio_prompts = true;
    bool match_segment_rms = true;
    float target_rms = 0.045f;
    double preroll_seconds = 2.0;
};

uint16_t read_u16(const std::vector<uint8_t>& data, size_t& offset) {
    if (offset + 2 > data.size()) throw std::runtime_error("Unexpected EOF");
    uint16_t value = (static_cast<uint16_t>(data[offset]) << 8) |
                     static_cast<uint16_t>(data[offset + 1]);
    offset += 2;
    return value;
}

uint32_t read_u32(const std::vector<uint8_t>& data, size_t& offset) {
    if (offset + 4 > data.size()) throw std::runtime_error("Unexpected EOF");
    uint32_t value = (static_cast<uint32_t>(data[offset]) << 24) |
                     (static_cast<uint32_t>(data[offset + 1]) << 16) |
                     (static_cast<uint32_t>(data[offset + 2]) << 8) |
                     static_cast<uint32_t>(data[offset + 3]);
    offset += 4;
    return value;
}

uint32_t read_vlq(const std::vector<uint8_t>& data, size_t& offset) {
    uint32_t value = 0;
    while (true) {
        if (offset >= data.size()) throw std::runtime_error("Unexpected EOF in VLQ");
        uint8_t byte = data[offset++];
        value = (value << 7) | (byte & 0x7f);
        if ((byte & 0x80) == 0) return value;
    }
}

std::string read_ascii(const std::vector<uint8_t>& data, size_t offset, size_t count) {
    if (offset + count > data.size()) throw std::runtime_error("Unexpected EOF in chunk tag");
    return std::string(reinterpret_cast<const char*>(data.data() + offset), count);
}

double tick_to_seconds(int tick, const std::vector<std::pair<int, int>>& tempo_events, int ppq) {
    std::vector<std::pair<int, int>> tempos = tempo_events;
    if (tempos.empty() || tempos.front().first != 0) {
        tempos.insert(tempos.begin(), {0, static_cast<int>(60000000.0 / kDefaultBpm)});
    }
    std::sort(tempos.begin(), tempos.end());
    double seconds = 0.0;
    int previous_tick = 0;
    int current_tempo_us = tempos.front().second;
    for (size_t i = 1; i < tempos.size(); ++i) {
        int tempo_tick = tempos[i].first;
        if (tick <= tempo_tick) break;
        seconds += (tempo_tick - previous_tick) * current_tempo_us / 1000000.0 / ppq;
        previous_tick = tempo_tick;
        current_tempo_us = tempos[i].second;
    }
    seconds += (tick - previous_tick) * current_tempo_us / 1000000.0 / ppq;
    return seconds;
}

MidiData parse_midi(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open MIDI file: " + path.string());
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    size_t offset = 0;
    if (read_ascii(data, offset, 4) != "MThd") {
        throw std::runtime_error("Not a Standard MIDI File: " + path.string());
    }
    offset += 4;
    uint32_t header_size = read_u32(data, offset);
    MidiData midi;
    midi.format = read_u16(data, offset);
    midi.tracks = read_u16(data, offset);
    midi.ppq = read_u16(data, offset);
    offset = 8 + header_size;

    std::vector<std::pair<int, int>> tempos;
    std::vector<std::tuple<int, int, int, int, int>> raw_notes;

    for (int track_index = 0; track_index < midi.tracks; ++track_index) {
        if (read_ascii(data, offset, 4) != "MTrk") {
            throw std::runtime_error("Expected MTrk chunk");
        }
        offset += 4;
        uint32_t track_size = read_u32(data, offset);
        if (offset + track_size > data.size()) throw std::runtime_error("Truncated MIDI track");
        std::vector<uint8_t> track(data.begin() + offset, data.begin() + offset + track_size);
        offset += track_size;

        size_t pos = 0;
        int absolute_tick = 0;
        int running_status = -1;
        std::map<std::pair<int, int>, std::vector<std::pair<int, int>>> active;
        while (pos < track.size()) {
            absolute_tick += static_cast<int>(read_vlq(track, pos));
            if (pos >= track.size()) break;
            int status = track[pos];
            if (status < 0x80) {
                if (running_status < 0) throw std::runtime_error("Running status without previous status");
                status = running_status;
            } else {
                ++pos;
                if (status < 0xf0) running_status = status;
            }

            if (status == 0xff) {
                if (pos >= track.size()) throw std::runtime_error("Bad meta event");
                int meta_type = track[pos++];
                int length = static_cast<int>(read_vlq(track, pos));
                if (pos + length > track.size()) throw std::runtime_error("Bad meta length");
                if (meta_type == 0x03) {
                    midi.track_name = std::string(track.begin() + pos, track.begin() + pos + length);
                } else if (meta_type == 0x51 && length == 3) {
                    int tempo_us = (track[pos] << 16) | (track[pos + 1] << 8) | track[pos + 2];
                    tempos.push_back({absolute_tick, tempo_us});
                } else if (meta_type == 0x58 && length >= 2) {
                    midi.time_signature = std::to_string(track[pos]) + "/" +
                                          std::to_string(1 << track[pos + 1]);
                } else if (meta_type == 0x2f) {
                    midi.end_tick = std::max(midi.end_tick, absolute_tick);
                    pos += length;
                    break;
                }
                pos += length;
                continue;
            }

            if (status == 0xf0 || status == 0xf7) {
                int length = static_cast<int>(read_vlq(track, pos));
                pos += length;
                continue;
            }

            int event_type = status & 0xf0;
            int channel = status & 0x0f;
            if (event_type == 0x80 || event_type == 0x90) {
                if (pos + 2 > track.size()) throw std::runtime_error("Bad note event");
                int pitch = track[pos++];
                int velocity = track[pos++];
                auto key = std::make_pair(channel, pitch);
                if (event_type == 0x90 && velocity > 0) {
                    active[key].push_back({absolute_tick, velocity});
                    midi.events.push_back({0.0, pitch, true});
                } else {
                    auto& starts = active[key];
                    if (!starts.empty()) {
                        auto [start_tick, start_velocity] = starts.front();
                        starts.erase(starts.begin());
                        raw_notes.push_back({start_tick, absolute_tick, pitch, start_velocity, channel});
                    }
                    midi.events.push_back({0.0, pitch, false});
                }
                midi.end_tick = std::max(midi.end_tick, absolute_tick);
            } else if (event_type == 0xa0 || event_type == 0xb0 || event_type == 0xe0) {
                pos += 2;
            } else if (event_type == 0xc0 || event_type == 0xd0) {
                pos += 1;
            } else {
                throw std::runtime_error("Unsupported MIDI status");
            }
        }
    }

    if (midi.track_name.empty()) midi.track_name = path.stem().string();
    int first_tempo = tempos.empty() ? static_cast<int>(60000000.0 / kDefaultBpm) : tempos.front().second;
    midi.bpm = 60000000.0 / first_tempo;
    midi.duration_seconds = tick_to_seconds(midi.end_tick, tempos, midi.ppq);

    midi.notes.clear();
    midi.events.clear();
    for (const auto& item : raw_notes) {
        MidiNote note;
        note.start_tick = std::get<0>(item);
        note.end_tick = std::get<1>(item);
        note.pitch = std::get<2>(item);
        note.velocity = std::get<3>(item);
        note.channel = std::get<4>(item);
        note.start_seconds = tick_to_seconds(note.start_tick, tempos, midi.ppq);
        note.end_seconds = tick_to_seconds(note.end_tick, tempos, midi.ppq);
        midi.notes.push_back(note);
        midi.events.push_back({note.start_seconds, note.pitch, true});
        midi.events.push_back({note.end_seconds, note.pitch, false});
    }
    std::sort(midi.notes.begin(), midi.notes.end(),
              [](const MidiNote& a, const MidiNote& b) {
                  if (a.start_seconds != b.start_seconds) return a.start_seconds < b.start_seconds;
                  return a.pitch < b.pitch;
              });
    std::sort(midi.events.begin(), midi.events.end(),
              [](const MidiEvent& a, const MidiEvent& b) {
                  if (a.time_seconds != b.time_seconds) return a.time_seconds < b.time_seconds;
                  if (a.on != b.on) return a.on && !b.on;
                  return a.pitch < b.pitch;
              });
    return midi;
}

std::string note_name(int pitch) {
    static const std::array<const char*, 12> names = {
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    return std::string(names[pitch % 12]) + std::to_string((pitch / 12) - 1);
}

double midi_frequency(int pitch) {
    return 440.0 * std::pow(2.0, (pitch - 69) / 12.0);
}

int segment_for_time(double time_seconds) {
    return std::clamp(static_cast<int>(time_seconds / 2.0), 0, 3);
}

std::array<float, 4> prompt_weights(double time_seconds) {
    constexpr double segment_seconds = 2.0;
    constexpr double transition_seconds = 0.20;
    int segment = segment_for_time(time_seconds);
    double local = time_seconds - segment * segment_seconds;
    int next = (segment + 1) % 4;
    std::array<float, 4> weights = {0.0f, 0.0f, 0.0f, 0.0f};
    if (local < segment_seconds - transition_seconds) {
        weights[segment] = 1.0f;
    } else {
        float amount = static_cast<float>(
            std::min(1.0, (local - (segment_seconds - transition_seconds)) / transition_seconds));
        weights[segment] = 1.0f - amount;
        weights[next] = amount;
    }
    return weights;
}

float blend_float(const std::array<float, 4>& weights, float Slot::*member) {
    float value = 0.0f;
    for (size_t i = 0; i < weights.size(); ++i) value += weights[i] * (kSlots[i].*member);
    return value;
}

int blend_top_k(const std::array<float, 4>& weights) {
    float value = 0.0f;
    for (size_t i = 0; i < weights.size(); ++i) value += weights[i] * kSlots[i].top_k;
    return static_cast<int>(std::lround(value));
}

std::vector<int> active_notes_at(const MidiData& midi, double time_seconds) {
    std::vector<int> result;
    for (const auto& note : midi.notes) {
        if (note.start_seconds <= time_seconds && time_seconds < note.end_seconds) {
            result.push_back(note.pitch);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

void normalize_audio(std::vector<float>& samples, float peak = 0.85f) {
    float max_abs = 0.0f;
    for (float v : samples) max_abs = std::max(max_abs, std::abs(v));
    if (max_abs <= 0.0f) return;
    float gain = peak / max_abs;
    for (float& v : samples) v *= gain;
}

void fade_edges(std::vector<float>& samples, double seconds = 0.01) {
    size_t count = std::min(samples.size() / 2, static_cast<size_t>(seconds * kSampleRate));
    if (count == 0) return;
    for (size_t i = 0; i < count; ++i) {
        float ramp = static_cast<float>(i) / static_cast<float>(count);
        samples[i] *= ramp;
        samples[samples.size() - 1 - i] *= ramp;
    }
}

void add_to(std::vector<float>& samples, int start, const std::vector<float>& tone) {
    if (start >= static_cast<int>(samples.size())) return;
    int end = std::min(static_cast<int>(samples.size()), start + static_cast<int>(tone.size()));
    for (int i = start; i < end; ++i) samples[i] += tone[i - start];
}

std::array<std::vector<int>, 4> segment_notes(const MidiData& midi) {
    std::array<std::vector<int>, 4> result;
    for (int segment = 0; segment < 4; ++segment) {
        result[segment] = active_notes_at(midi, segment * 2.0);
    }
    return result;
}

std::vector<float> synth_piano_prompt(const MidiData& midi) {
    std::vector<float> samples(static_cast<size_t>(midi.duration_seconds * kSampleRate), 0.0f);
    for (const auto& note : midi.notes) {
        int start = static_cast<int>(std::lround(note.start_seconds * kSampleRate));
        int end = std::min(static_cast<int>(samples.size()),
                           static_cast<int>(std::lround(note.end_seconds * kSampleRate)));
        for (int i = start; i < end; ++i) {
            double t = (i - start) / static_cast<double>(kSampleRate);
            double f = midi_frequency(note.pitch);
            double env = std::exp(-3.2 * t);
            double tone = (std::sin(2 * kPi * f * t) +
                           0.42 * std::sin(2 * kPi * f * 2.01 * t) +
                           0.18 * std::sin(2 * kPi * f * 3.0 * t)) * env;
            samples[i] += static_cast<float>(0.18 * tone);
        }
    }
    fade_edges(samples);
    normalize_audio(samples);
    return samples;
}

std::vector<float> synth_chiptune_prompt(const MidiData& midi) {
    std::vector<float> samples(static_cast<size_t>(midi.duration_seconds * kSampleRate), 0.0f);
    auto notes = segment_notes(midi);
    constexpr double step_seconds = 0.125;
    int step_count = static_cast<int>(std::lround(midi.duration_seconds / step_seconds));
    for (int step = 0; step < step_count; ++step) {
        double start_seconds = step * step_seconds;
        int segment = segment_for_time(start_seconds);
        if (notes[segment].empty()) continue;
        int pitch = notes[segment][step % notes[segment].size()] + 24;
        double f = midi_frequency(pitch);
        int start = static_cast<int>(std::lround(start_seconds * kSampleRate));
        int length = static_cast<int>(std::lround(step_seconds * 0.82 * kSampleRate));
        for (int j = 0; j < length && start + j < static_cast<int>(samples.size()); ++j) {
            double t = j / static_cast<double>(kSampleRate);
            double square = std::sin(2 * kPi * f * t) >= 0.0 ? 1.0 : -1.0;
            double env = std::min(1.0, t / 0.006) * std::exp(-5.5 * t);
            samples[start + j] += static_cast<float>(0.25 * square * env);
        }
    }
    fade_edges(samples);
    normalize_audio(samples);
    return samples;
}

std::vector<float> synth_808_prompt(const MidiData& midi) {
    std::vector<float> samples(static_cast<size_t>(midi.duration_seconds * kSampleRate), 0.0f);
    auto notes = segment_notes(midi);
    std::mt19937 rng(2405);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    for (double beat = 0.0; beat < midi.duration_seconds; beat += 0.5) {
        int segment = segment_for_time(beat);
        int root = notes[segment].empty() ? 36 : notes[segment].front() - 12;
        int length = static_cast<int>(std::lround(0.36 * kSampleRate));
        std::vector<float> tone(length);
        double phase = 0.0;
        for (int i = 0; i < length; ++i) {
            double t = i / static_cast<double>(kSampleRate);
            double f = midi_frequency(root) * (1.7 - 0.55 * std::min(1.0, t / 0.16));
            phase += 2 * kPi * f / kSampleRate;
            tone[i] = static_cast<float>(0.55 * std::sin(phase) * std::exp(-8.0 * t));
        }
        add_to(samples, static_cast<int>(std::lround(beat * kSampleRate)), tone);
    }
    for (double snare = 1.0; snare < midi.duration_seconds; snare += 2.0) {
        int length = static_cast<int>(std::lround(0.18 * kSampleRate));
        std::vector<float> tone(length);
        for (int i = 0; i < length; ++i) {
            double t = i / static_cast<double>(kSampleRate);
            tone[i] = static_cast<float>(0.18 * noise(rng) * std::exp(-22.0 * t));
        }
        add_to(samples, static_cast<int>(std::lround(snare * kSampleRate)), tone);
    }
    for (double hat = 0.0; hat < midi.duration_seconds; hat += 0.125) {
        int length = static_cast<int>(std::lround(0.035 * kSampleRate));
        std::vector<float> tone(length);
        for (int i = 0; i < length; ++i) {
            double t = i / static_cast<double>(kSampleRate);
            tone[i] = static_cast<float>(0.045 * noise(rng) *
                                         std::sin(2 * kPi * 7600.0 * t) *
                                         std::exp(-70.0 * t));
        }
        add_to(samples, static_cast<int>(std::lround(hat * kSampleRate)), tone);
    }
    for (float& sample : samples) sample = std::tanh(sample * 2.8f);
    fade_edges(samples);
    normalize_audio(samples);
    return samples;
}

std::vector<float> synth_pad_prompt(const MidiData& midi) {
    std::vector<float> samples(static_cast<size_t>(midi.duration_seconds * kSampleRate), 0.0f);
    for (const auto& note : midi.notes) {
        int start = static_cast<int>(std::lround(note.start_seconds * kSampleRate));
        int end = std::min(static_cast<int>(samples.size()),
                           static_cast<int>(std::lround(note.end_seconds * kSampleRate)));
        for (int i = start; i < end; ++i) {
            double t = (i - start) / static_cast<double>(kSampleRate);
            double dur = note.end_seconds - note.start_seconds;
            double f = midi_frequency(note.pitch + 12);
            double attack = std::min(1.0, t / 0.65);
            double release = std::min(1.0, std::max(0.0, (dur - t) / 0.45));
            double env = std::min(attack, release);
            double tone = 0.58 * std::sin(2 * kPi * f * t) +
                          0.32 * std::sin(2 * kPi * f * 1.5 * t) +
                          0.22 * std::sin(2 * kPi * f * 2.01 * t);
            samples[i] += static_cast<float>(0.12 * tone * env);
        }
    }
    int delay = static_cast<int>(std::lround(0.19 * kSampleRate));
    for (int i = static_cast<int>(samples.size()) - 1; i >= delay; --i) {
        samples[i] += 0.34f * samples[i - delay];
    }
    fade_edges(samples);
    normalize_audio(samples);
    return samples;
}

std::array<std::vector<float>, 4> build_audio_prompts(const MidiData& midi) {
    return {synth_piano_prompt(midi), synth_chiptune_prompt(midi),
            synth_808_prompt(midi), synth_pad_prompt(midi)};
}

bool write_wav(const std::filesystem::path& path,
               const std::vector<float>& interleaved,
               int sample_rate,
               int num_channels) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    uint32_t num_frames = static_cast<uint32_t>(interleaved.size()) / num_channels;
    uint16_t bits_per_sample = 32;
    uint16_t block_align = num_channels * (bits_per_sample / 8);
    uint32_t byte_rate = sample_rate * block_align;
    uint32_t data_size = num_frames * block_align;
    uint32_t chunk_size = 36 + data_size;
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write("RIFF", 4);
    f.write(reinterpret_cast<const char*>(&chunk_size), 4);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    uint32_t subchunk1_size = 16;
    uint16_t audio_format = 3;
    uint16_t channels = static_cast<uint16_t>(num_channels);
    uint32_t sr = static_cast<uint32_t>(sample_rate);
    f.write(reinterpret_cast<const char*>(&subchunk1_size), 4);
    f.write(reinterpret_cast<const char*>(&audio_format), 2);
    f.write(reinterpret_cast<const char*>(&channels), 2);
    f.write(reinterpret_cast<const char*>(&sr), 4);
    f.write(reinterpret_cast<const char*>(&byte_rate), 4);
    f.write(reinterpret_cast<const char*>(&block_align), 2);
    f.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
    f.write("data", 4);
    f.write(reinterpret_cast<const char*>(&data_size), 4);
    f.write(reinterpret_cast<const char*>(interleaved.data()),
            interleaved.size() * sizeof(float));
    return f.good();
}

std::vector<float> to_stereo_interleaved(const std::vector<float>& mono) {
    std::vector<float> interleaved;
    interleaved.reserve(mono.size() * 2);
    for (float v : mono) {
        interleaved.push_back(v);
        interleaved.push_back(v);
    }
    return interleaved;
}

float rms_segment(const std::vector<float>& interleaved, int start_frame, int end_frame) {
    double sum = 0.0;
    int count = 0;
    for (int frame = start_frame; frame < end_frame; ++frame) {
        float left = interleaved[frame * 2];
        float right = interleaved[frame * 2 + 1];
        float mono = 0.5f * (left + right);
        sum += mono * mono;
        ++count;
    }
    return count > 0 ? static_cast<float>(std::sqrt(sum / count)) : 0.0f;
}

SegmentMetrics metrics_segment(const std::vector<float>& interleaved, int start_frame, int end_frame) {
    SegmentMetrics metrics;
    double sum = 0.0;
    int count = 0;
    for (int frame = start_frame; frame < end_frame; ++frame) {
        float left = interleaved[frame * 2];
        float right = interleaved[frame * 2 + 1];
        float mono = 0.5f * (left + right);
        metrics.peak = std::max(metrics.peak, std::abs(mono));
        sum += mono * mono;
        ++count;
    }
    metrics.rms = count > 0 ? static_cast<float>(std::sqrt(sum / count)) : 0.0f;
    return metrics;
}

void match_segment_rms(std::vector<float>& interleaved, double duration_seconds, float target_rms) {
    int total_frames = static_cast<int>(interleaved.size() / 2);
    for (int segment = 0; segment < 4; ++segment) {
        int start = static_cast<int>(std::lround(segment * 2.0 * kSampleRate));
        int end = std::min(total_frames, static_cast<int>(std::lround((segment + 1) * 2.0 * kSampleRate)));
        if (segment * 2.0 >= duration_seconds) break;
        float rms = rms_segment(interleaved, start, end);
        float gain = rms > 0.000001f ? std::min(8.0f, target_rms / rms) : 1.0f;
        for (int frame = start; frame < end; ++frame) {
            interleaved[frame * 2] *= gain;
            interleaved[frame * 2 + 1] *= gain;
        }
    }
    float peak = 0.0f;
    for (float v : interleaved) peak = std::max(peak, std::abs(v));
    float target_peak = std::pow(10.0f, -1.0f / 20.0f);
    if (peak > target_peak) {
        float gain = target_peak / peak;
        for (float& v : interleaved) v *= gain;
    }
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

bool write_report(const std::filesystem::path& path,
                  const RenderConfig& config,
                  const MidiData& midi,
                  const std::vector<float>& interleaved,
                  const std::string& model_path,
                  double elapsed_seconds) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out) return false;
    int total_frames = static_cast<int>(interleaved.size() / 2);
    float peak = 0.0f;
    double sum = 0.0;
    for (float v : interleaved) {
        peak = std::max(peak, std::abs(v));
        sum += v * v;
    }
    double rms = interleaved.empty() ? 0.0 : std::sqrt(sum / interleaved.size());

    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"schema\": \"mrt-cpp-midi-control-matrix-report-v1\",\n";
    out << "  \"output_wav\": \"" << json_escape(config.output_path.string()) << "\",\n";
    out << "  \"midi_path\": \"" << json_escape(config.midi_path.string()) << "\",\n";
    out << "  \"model\": \"" << json_escape(config.model_name) << "\",\n";
    out << "  \"model_path\": \"" << json_escape(model_path) << "\",\n";
    out << "  \"embedding_source\": \"" << (config.use_audio_prompts ? "audio" : "text") << "\",\n";
    out << "  \"duration_seconds\": " << (static_cast<double>(total_frames) / kSampleRate) << ",\n";
    out << "  \"expected_duration_seconds\": " << midi.duration_seconds << ",\n";
    out << "  \"sample_rate\": " << kSampleRate << ",\n";
    out << "  \"channels\": 2,\n";
    out << "  \"frames_25hz\": " << static_cast<int>(std::lround(midi.duration_seconds * kFps)) << ",\n";
    out << "  \"chunk_samples\": " << kFrameSamples << ",\n";
    out << "  \"peak\": " << peak << ",\n";
    out << "  \"rms\": " << rms << ",\n";
    out << "  \"non_silent\": " << ((peak > 0.0001f && rms > 0.00001) ? "true" : "false") << ",\n";
    out << "  \"midi\": {\n";
    out << "    \"track_name\": \"" << json_escape(midi.track_name) << "\",\n";
    out << "    \"format\": " << midi.format << ",\n";
    out << "    \"tracks\": " << midi.tracks << ",\n";
    out << "    \"ppq\": " << midi.ppq << ",\n";
    out << "    \"bpm\": " << midi.bpm << ",\n";
    out << "    \"time_signature\": \"" << json_escape(midi.time_signature) << "\",\n";
    out << "    \"notes\": " << midi.notes.size() << "\n";
    out << "  },\n";
    out << "  \"control_roles\": {\n";
    out << "    \"primary\": \"prompt embedding mix via reblend_musiccoca_tokens\",\n";
    out << "    \"secondary\": \"cfg_musiccoca / cfg_notes / cfg_drums\",\n";
    out << "    \"expression\": \"temperature\",\n";
    out << "    \"exploration\": \"top_k\",\n";
    out << "    \"stability\": \"fixed 25 Hz frames / 1920 sample chunks\",\n";
    out << "    \"midi\": \"MLXEngine::set_note_on/off -> MidiNoteTracker -> generate_frame\"\n";
    out << "  },\n";
    out << "  \"segments\": [\n";
    for (int segment = 0; segment < 4; ++segment) {
        int start = static_cast<int>(std::lround(segment * 2.0 * kSampleRate));
        int end = std::min(total_frames, static_cast<int>(std::lround((segment + 1) * 2.0 * kSampleRate)));
        SegmentMetrics metrics = metrics_segment(interleaved, start, end);
        auto notes = active_notes_at(midi, segment * 2.0);
        auto weights = prompt_weights(segment * 2.0);
        out << "    {\n";
        out << "      \"segment\": " << (segment + 1) << ",\n";
        out << "      \"slot_id\": \"" << kSlots[segment].id << "\",\n";
        out << "      \"slot_label\": \"" << kSlots[segment].label << "\",\n";
        out << "      \"weight\": " << weights[segment] << ",\n";
        out << "      \"cfg_musiccoca\": " << kSlots[segment].cfg_musiccoca << ",\n";
        out << "      \"cfg_notes\": " << kSlots[segment].cfg_notes << ",\n";
        out << "      \"cfg_drums\": " << kSlots[segment].cfg_drums << ",\n";
        out << "      \"temperature\": " << kSlots[segment].temperature << ",\n";
        out << "      \"top_k\": " << kSlots[segment].top_k << ",\n";
        out << "      \"rms\": " << metrics.rms << ",\n";
        out << "      \"peak\": " << metrics.peak << ",\n";
        out << "      \"active_note_names\": [";
        for (size_t i = 0; i < notes.size(); ++i) {
            if (i) out << ", ";
            out << "\"" << note_name(notes[i]) << "\"";
        }
        out << "]\n";
        out << "    }" << (segment == 3 ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"elapsed_seconds\": " << elapsed_seconds << "\n";
    out << "}\n";
    return out.good();
}

void wait_for_prompts(MLXEngine& engine) {
    while (engine.get_text_encoder_status() == 1 ||
           engine.get_quantizer_status() == 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    for (int i = 0; i < static_cast<int>(kSlots.size()); ++i) {
        while (engine.get_prompt_status(i) == 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --midi PATH              MIDI file (default: assets/Am-Minor Prog 01 (i-VI-v-iv).mid)\n"
        "  --output PATH            Output WAV path\n"
        "  --report PATH            Output JSON report path\n"
        "  --model NAME             Model folder under Magenta models (default: mrt2_small)\n"
        "  --resources PATH         Resource dir containing musiccoca/\n"
        "  --text-prompts           Use text prompts instead of MIDI-derived audio prompt embeddings\n"
        "  --no-match-rms           Do not RMS-match four 2-second sections\n"
        "  --target-rms VALUE       Target RMS when matching sections (default: 0.045)\n",
        argv0);
}

bool parse_args(int argc, char** argv, RenderConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--midi") {
            config.midi_path = need_value("--midi");
        } else if (arg == "--output" || arg == "-o") {
            config.output_path = need_value("--output");
        } else if (arg == "--report") {
            config.report_path = need_value("--report");
        } else if (arg == "--model") {
            config.model_name = need_value("--model");
        } else if (arg == "--resources") {
            config.resource_dir = need_value("--resources");
        } else if (arg == "--audio-prompt-dir") {
            config.audio_prompt_dir = need_value("--audio-prompt-dir");
        } else if (arg == "--text-prompts") {
            config.use_audio_prompts = false;
        } else if (arg == "--no-match-rms") {
            config.match_segment_rms = false;
        } else if (arg == "--target-rms") {
            config.target_rms = std::stof(need_value("--target-rms"));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    RenderConfig config;
    if (!parse_args(argc, argv, config)) return 1;

    try {
        MidiData midi = parse_midi(config.midi_path);
        int frame_count = static_cast<int>(std::lround(midi.duration_seconds * kFps));
        std::string model_dir = magentart::paths::get_models_dir() + "/" + config.model_name;
        std::string mlxfn_path = magentart::paths::find_mlxfn_in_dir(model_dir);
        if (mlxfn_path.empty()) {
            std::fprintf(stderr, "Could not find .mlxfn in %s\n", model_dir.c_str());
            return 1;
        }

        MLXEngine engine;
        if (!engine.init_assets(config.resource_dir.c_str(), config.model_subfolder.c_str())) {
            std::fprintf(stderr, "Failed to init resources from %s\n", config.resource_dir.c_str());
            return 1;
        }
        if (!engine.load_model(mlxfn_path.c_str())) {
            std::fprintf(stderr, "Failed to load model %s\n", mlxfn_path.c_str());
            return 1;
        }

        if (config.use_audio_prompts) {
            auto prompt_audio = build_audio_prompts(midi);
            std::filesystem::create_directories(config.audio_prompt_dir);
            for (int i = 0; i < 4; ++i) {
                auto audio_path = config.audio_prompt_dir /
                                  (std::to_string(i + 1) + "_" + kSlots[i].id + ".wav");
                write_wav(audio_path, to_stereo_interleaved(prompt_audio[i]), kSampleRate, 2);
                engine.set_audio_prompt_samples(i, audio_path.filename().string(),
                                                prompt_audio[i].data(), prompt_audio[i].size());
            }
            wait_for_prompts(engine);
            std::array<float, 6> initial_weights = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            if (!engine.reblend_musiccoca_tokens(initial_weights.data(), static_cast<int>(initial_weights.size()))) {
                std::fprintf(stderr, "Initial audio prompt reblend failed\n");
                return 1;
            }
        } else {
            std::vector<std::string> prompts;
            std::vector<float> initial_weights;
            for (int i = 0; i < 4; ++i) {
                prompts.push_back(kSlots[i].prompt);
                initial_weights.push_back(i == 0 ? 1.0f : 0.0f);
            }
            engine.set_text_prompts(prompts, initial_weights);
            wait_for_prompts(engine);
        }

        engine.set_onset_mode(1);
        engine.set_unmask_width(127);
        engine.set_drumless(false);
        engine.reset_state();

        // Match the official live-MIDI path during preroll too, so the model
        // is already conditioned by the first held chord before audible output.
        for (int pitch : active_notes_at(midi, 0.0)) {
            engine.set_note_on(pitch);
        }

        auto start_time = std::chrono::steady_clock::now();
        std::vector<float> L(kFrameSamples), R(kFrameSamples);

        // Warm up model state while holding the first prompt, without writing audio.
        int preroll_frames = static_cast<int>(std::lround(config.preroll_seconds * kFps));
        for (int i = 0; i < preroll_frames; ++i) {
            engine.generate_frame(L.data(), R.data());
        }

        std::vector<float> interleaved;
        interleaved.reserve(static_cast<size_t>(frame_count) * kFrameSamples * 2);
        size_t next_event = 0;
        for (int frame = 0; frame < frame_count; ++frame) {
            double time_seconds = frame * kFrameSeconds;
            while (next_event < midi.events.size() &&
                   midi.events[next_event].time_seconds <= time_seconds + 0.000001) {
                if (midi.events[next_event].on) {
                    engine.set_note_on(midi.events[next_event].pitch);
                } else {
                    engine.set_note_off(midi.events[next_event].pitch);
                }
                ++next_event;
            }

            auto weights4 = prompt_weights(time_seconds);
            std::array<float, 6> weights = {
                weights4[0], weights4[1], weights4[2], weights4[3], 0.0f, 0.0f};
            engine.reblend_musiccoca_tokens(weights.data(), static_cast<int>(weights.size()));
            engine.set_cfg_musiccoca(blend_float(weights4, &Slot::cfg_musiccoca));
            engine.set_cfg_notes(blend_float(weights4, &Slot::cfg_notes));
            engine.set_cfg_drums(blend_float(weights4, &Slot::cfg_drums));
            engine.set_temperature(blend_float(weights4, &Slot::temperature));
            engine.set_top_k(blend_top_k(weights4));

            if (!engine.generate_frame(L.data(), R.data())) {
                std::fprintf(stderr, "generate_frame failed at frame %d\n", frame);
                return 1;
            }
            for (size_t i = 0; i < kFrameSamples; ++i) {
                interleaved.push_back(L[i]);
                interleaved.push_back(R[i]);
            }
            if ((frame + 1) % 50 == 0) {
                std::printf("frame %d/%d\n", frame + 1, frame_count);
            }
        }
        while (next_event < midi.events.size()) {
            if (midi.events[next_event].on) {
                engine.set_note_on(midi.events[next_event].pitch);
            } else {
                engine.set_note_off(midi.events[next_event].pitch);
            }
            ++next_event;
        }

        if (config.match_segment_rms) {
            match_segment_rms(interleaved, midi.duration_seconds, config.target_rms);
        }

        auto end_time = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(end_time - start_time).count();
        if (!write_wav(config.output_path, interleaved, kSampleRate, 2)) {
            std::fprintf(stderr, "Failed to write %s\n", config.output_path.string().c_str());
            return 1;
        }
        if (!write_report(config.report_path, config, midi, interleaved, mlxfn_path, elapsed)) {
            std::fprintf(stderr, "Failed to write %s\n", config.report_path.string().c_str());
            return 1;
        }
        std::printf("Wrote %s\n", config.output_path.string().c_str());
        std::printf("Wrote %s\n", config.report_path.string().c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
