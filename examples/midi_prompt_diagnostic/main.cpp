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
constexpr int kWeightFrameRate = 60;
constexpr double kFrameSeconds = 1.0 / kFps;
constexpr double kWeightFrameSeconds = 1.0 / kWeightFrameRate;
constexpr double kDefaultBpm = 120.0;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr std::array<double, 4> kMeterQuarterNotePeriods = {
    4.0,  // 4/4
    3.0,  // 3/4
    3.0,  // 6/8 as six eighth notes
    4.0,  // 4/4
};
constexpr std::array<double, 4> kMeterPhaseOffsets = {0.00, 0.00, 0.00, 0.25};
constexpr std::array<double, 4> kMacroCyclesPerReference = {1.0, 1.5, 2.5, 3.5};
constexpr std::array<double, 4> kMacroPhaseOffsets = {0.00, 0.23, 0.47, 0.71};

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
    std::string guide_kind;
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
        "piano",
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
        "chiptune",
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
        "drums808",
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
        "pad",
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
    std::array<Slot, 4> slots = kSlots;
    std::string profile = "clear_extreme";
    std::string weight_mode = "sequential";
    std::string control_mode = "slot_blend";
    int solo_slot = 0;
    double bpm = kDefaultBpm;
    double duration_seconds = 0.0;
    double segment_seconds = 2.0;
    double transition_seconds = 0.20;
    bool use_audio_prompts = true;
    bool match_segment_rms = true;
    float target_rms = 0.045f;
    bool stabilize_window_rms = false;
    float min_window_rms = 0.002f;
    float max_window_gain = 256.0f;
    double preroll_seconds = 2.0;
    std::filesystem::path weights_path;
    std::filesystem::path prompt_library_path;
    double prompt_page_seconds = 4.0;
    std::vector<Slot> prompt_library;
    int batch_variant = 0;
    double macro_phase_offset = 0.0;
    double macro_reference_seconds = 128.0;
    double midi_refresh_seconds = 0.0;
    std::string midi_mode = "scheduled";
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
        seconds += static_cast<double>(tempo_tick - previous_tick) *
                   static_cast<double>(current_tempo_us) / 1000000.0 / ppq;
        previous_tick = tempo_tick;
        current_tempo_us = tempos[i].second;
    }
    seconds += static_cast<double>(tick - previous_tick) *
               static_cast<double>(current_tempo_us) / 1000000.0 / ppq;
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

MidiData repeat_midi_to_duration(const MidiData& source, double duration_seconds) {
    if (duration_seconds <= 0.0 ||
        source.duration_seconds <= 0.0 ||
        std::abs(duration_seconds - source.duration_seconds) < 0.001) {
        return source;
    }

    MidiData midi = source;
    midi.duration_seconds = duration_seconds;
    midi.notes.clear();
    midi.events.clear();

    int cycles = static_cast<int>(std::ceil(duration_seconds / source.duration_seconds));
    for (int cycle = 0; cycle < cycles; ++cycle) {
        double offset_seconds = cycle * source.duration_seconds;
        for (const auto& src_note : source.notes) {
            double start = src_note.start_seconds + offset_seconds;
            double end = src_note.end_seconds + offset_seconds;
            if (start >= duration_seconds) continue;
            end = std::min(end, duration_seconds);
            if (end <= start) continue;

            MidiNote note = src_note;
            note.start_seconds = start;
            note.end_seconds = end;
            note.start_tick = src_note.start_tick + cycle * source.end_tick;
            note.end_tick = src_note.end_tick + cycle * source.end_tick;
            midi.notes.push_back(note);
            midi.events.push_back({note.start_seconds, note.pitch, true});
            midi.events.push_back({note.end_seconds, note.pitch, false});
        }
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

bool apply_profile(RenderConfig& config) {
    if (config.profile == "clear_extreme" || config.profile == "default") {
        config.slots = kSlots;
        return true;
    }
    if (config.profile == "microcinematic_footwork") {
        config.slots = {{
            {
                "footwork_grid",
                "Microcinematic Footwork",
                "original microcinematic footwork loop material, 160 BPM feel, rapid footwork percussion cells, dry clipped snares, short sub hits, microscopic glitch cuts, dark film-sound tension, no vocals, no long intro",
                "rapid footwork kick, snare, hat, and glitch guide pattern",
                "footwork",
                1.28f,
                192,
                7.2f,
                2.8f,
                7.4f,
            },
            {
                "sub_trap_stabs",
                "Sub Trap Stabs",
                "original subpixel trap loop material, heavy 808 bass stabs, half-time low-end punctuation, sparse swung hats, chopped minor chord accents, clear downbeat, no vocals, no EDM drop",
                "808 sub bass and sparse trap drum guide from MIDI roots",
                "subbass",
                1.12f,
                144,
                6.8f,
                3.2f,
                6.4f,
            },
            {
                "granular_metal_cuts",
                "Granular Metal Cuts",
                "original glitch percussion loop material, metallic micro-edits, granular fragments, short digital cuts, tiny reversed impacts, high contrast stereo space, tempo-stable, no melody lead",
                "metallic FM hits and buffer-cut noise guide",
                "metallic",
                1.42f,
                220,
                6.6f,
                2.4f,
                3.5f,
            },
            {
                "noir_string_pressure",
                "Noir String Pressure",
                "original cinematic noir drone loop material, tense low strings, distant choir pad, slow pressure rise, wide reverb field, harmonically stable minor center, no drums, no vocals",
                "slow cinematic pad and string guide from MIDI chords",
                "pad",
                0.66f,
                40,
                7.0f,
                6.3f,
                0.0f,
            },
        }};
        return true;
    }
    if (config.profile == "negative_space_club") {
        config.slots = {{
            {
                "dry_minimal_pulse",
                "Dry Minimal Pulse",
                "original negative space club loop material, 128 BPM feel, dry four-on-floor kick, reduced pulse, tiny timbral shifts, restrained percussion, no vocals, no big drop",
                "minimal techno kick and click guide",
                "minimal",
                0.82f,
                54,
                6.0f,
                2.6f,
                6.8f,
            },
            {
                "dub_chord_stabs",
                "Dub Chord Stabs",
                "original dub techno chord loop material, short filtered minor chord stabs, tape delay space, stable low-end pulse, sparse percussion, clear downbeat, no vocals",
                "short piano-like chord stab guide from MIDI voicings",
                "piano",
                0.92f,
                72,
                6.8f,
                5.8f,
                1.8f,
            },
            {
                "sub_pressure",
                "Sub Pressure",
                "original cinematic bass minimalism loop material, deep sub bass pressure, slow sidechain-like movement, dark room tone, simple pulse, no lead melody, no vocals",
                "deep sine sub guide from MIDI roots",
                "subbass",
                0.78f,
                48,
                6.4f,
                3.6f,
                2.8f,
            },
            {
                "air_field",
                "Air Field",
                "original ambient club field loop material, airy granular noise, long reverb tail, slow stereo motion, low pressure atmosphere, no drums, no vocals, no tempo drift",
                "soft pad and noise field guide",
                "pad",
                0.58f,
                28,
                6.6f,
                5.8f,
                0.0f,
            },
        }};
        return true;
    }
    if (config.profile == "glass_trap_pressure") {
        config.slots = {{
            {
                "glass_mallets",
                "Glass Mallets",
                "original glass trap pressure loop material, bright glass mallet arpeggios, sharp transients, minor chord sparkle, tempo-stable, clear downbeat, no vocals",
                "bright glass and metallic FM mallet guide from MIDI notes",
                "metallic",
                1.20f,
                180,
                7.0f,
                4.8f,
                0.4f,
            },
            {
                "808_trap_weight",
                "808 Trap Weight",
                "original trap weight loop material, distorted 808 sub, punchy kick, clipped snare, swung hi-hats, low-end punctuation, no vocals, no long intro",
                "808 drum and sub guide from MIDI roots",
                "drums808",
                1.16f,
                156,
                6.8f,
                2.8f,
                7.2f,
            },
            {
                "microcut_texture",
                "Microcut Texture",
                "original glitch texture loop material, short buffer edits, sliced noise bursts, digital chirps, broken rhythmic accents, high contrast, no vocals, no sustained pad",
                "rapid chiptune and glitch guide",
                "chiptune",
                1.46f,
                230,
                6.5f,
                2.5f,
                2.4f,
            },
            {
                "cinematic_afterglow",
                "Cinematic Afterglow",
                "original cinematic afterglow loop material, warm strings, distant choir, soft tape noise, wide late-night room, minor harmonic center, no drums, no vocals",
                "slow string and choir pad guide from MIDI chords",
                "pad",
                0.62f,
                36,
                6.9f,
                6.2f,
                0.0f,
            },
        }};
        return true;
    }
    if (config.profile == "metallic_ambient_bounce") {
        config.slots = {{
            {
                "metallic_bounce",
                "Metallic Ambient Bounce",
                "original metallic ambient bounce loop material, soft syncopated percussion, tuned metal taps, gentle club bounce, spacious and precise, no vocals, no EDM drop",
                "metallic tap guide with soft pulse",
                "metallic",
                1.04f,
                112,
                6.7f,
                4.2f,
                3.8f,
            },
            {
                "granular_air",
                "Granular Air",
                "original granular ambient loop material, drifting noise grains, soft reversed fragments, slow stereo motion, low-pressure atmosphere, no drums, no vocals",
                "pad and granular air guide",
                "pad",
                0.60f,
                34,
                6.7f,
                5.6f,
                0.0f,
            },
            {
                "minimal_low_end",
                "Minimal Low End",
                "original low-end minimal loop material, quiet sub pulse, restrained kick, small percussion clicks, dark negative space, tempo-stable, no vocals",
                "minimal kick and sub pulse guide",
                "minimal",
                0.78f,
                52,
                6.2f,
                3.0f,
                5.4f,
            },
            {
                "glass_chord_cloud",
                "Glass Chord Cloud",
                "original glass chord cloud loop material, shimmering mallets, sustained minor chord haze, soft resonance, no drums, no vocals, seamless loop boundary",
                "glass mallet and chord shimmer guide",
                "chiptune",
                0.92f,
                96,
                6.8f,
                5.4f,
                0.0f,
            },
        }};
        return true;
    }
    if (config.profile == "sustained_synth_textures") {
        config.slots = {{
            {
                "low_voltage_fog",
                "Low Voltage Fog",
                "Create an original sustained synthesizer texture called Low Voltage Fog. A continuous low-register modular VCO drone with slow phase beating, slight pitch drift, and a dark ladder low-pass filter opening over time. Add warm saturation, tape delay feedback, and a wide stable reverb tail. Cinematic, heavy, evolving, no clear melody. Avoid drums, vocals, arpeggios, EDM drops, bright pop chords, and abrupt endings.",
                "low modular VCO drone with phase beating, filter motion, tape delay, and reverb",
                "analog_drone",
                0.66f,
                48,
                7.2f,
                5.8f,
                0.0f,
            },
            {
                "spectral_glass_bloom",
                "Spectral Glass Bloom",
                "Create an original sustained synthesizer texture called Spectral Glass Bloom. A bright high-mid wavetable pad with slow spectral freeze thawing, subtle phase smear, and glassy harmonic drift. Add shimmer reverb, gentle chorus, a resonant high-pass air layer, and very slow stereo widening. Luminous, cinematic, floating, continuously evolving. Avoid drums, vocals, lead melody, fast arpeggios, EDM drops, and abrupt endings.",
                "bright wavetable-like pad with spectral shimmer, chorus, and slow stereo drift",
                "spectral_pad",
                0.72f,
                72,
                7.0f,
                5.6f,
                0.0f,
            },
            {
                "carbon_cinema_noise",
                "Carbon Cinema Noise",
                "Create an original sustained synthesizer texture called Carbon Cinema Noise. A dark cinematic noise bed with filtered broadband noise, distant sub pressure, slow band-pass sweep, and occasional resonant bloom without rhythm. Add convolution reverb, low tape saturation, and a huge distant room tail. Tense, smoky, physical, continuous. Avoid drums, vocals, melody lead, bright chords, riser clichés, EDM drops, and abrupt endings.",
                "dark filtered noise bed with sub pressure, resonant sweeps, and convolution-like tail",
                "cinema_noise",
                0.88f,
                96,
                7.4f,
                3.4f,
                0.0f,
            },
            {
                "buffer_freeze_dust",
                "Buffer Freeze Dust",
                "Create an original sustained synthesizer texture called Buffer Freeze Dust. A glitchy granular sustain made from frozen buffers, bit-depth erosion, small digital crackles, and slowly changing grain density. Add spectral smear, short reverse delay, resonant notch movement, and a wide unstable stereo field. Fragile, electric, textural, continuous. Avoid drums, vocals, obvious melody, fast arpeggios, EDM drops, and clean polished pop chords.",
                "granular frozen buffer texture with bitcrushed edges, reverse delay, and notch movement",
                "granular_glitch",
                1.05f,
                128,
                6.8f,
                2.8f,
                0.0f,
            },
        }};
        return true;
    }
    if (config.profile == "dirty_cinematic_ambient_cm9") {
        config.slots = {{
            {
                "dirty_voltage_haze",
                "Dirty Voltage Haze",
                "Create an original sustained synthesizer texture called Dirty Voltage Haze. A continuous Cm9 low-register modular VCO drone held for 64 bars at 120 BPM, dirty and cinematic, with slow non-rhythmic phase drift, dark ladder low-pass movement, and saturated analog noise in the background. Add tape compression, soft overdrive, filtered reverb, and a wide stable tail. Keep the harmony present and sustained from start to finish. Avoid drums, percussion, kick, snare, hats, clicks, arpeggios, sequencer patterns, rhythmic pulses, tremolo, delay taps, vocals, lead melody, drops, fade-outs, and silence.",
                "dirty Cm9 modular VCO drone with tape saturation and slow filter drift",
                "analog_drone",
                0.56f,
                42,
                5.8f,
                5.7f,
                0.0f,
            },
            {
                "carbon_cinema_floor",
                "Carbon Cinema Floor",
                "Create an original sustained synthesizer texture called Carbon Cinema Floor. A continuous Cm9 cinematic noise bed with filtered broadband noise, low sub pressure, smoky texture, and slow band-pass color changes without impacts or rhythm. Add convolution space, dark saturation, soft spectral blur, and a huge distant-room tail. The sound should be dirty, ambient, heavy, and evolving as one long held environment. Avoid drums, percussion, impacts, risers, hits, arpeggios, sequencer patterns, rhythmic pulses, delay taps, vocals, lead melody, drops, fade-outs, and silence.",
                "dirty cinematic Cm9 noise floor with sub pressure and convolution space",
                "cinema_noise",
                0.64f,
                58,
                6.1f,
                5.8f,
                0.0f,
            },
            {
                "frozen_glitch_sheet",
                "Frozen Glitch Sheet",
                "Create an original sustained synthesizer texture called Frozen Glitch Sheet. A continuous Cm9 frozen-buffer synth sheet with glitch texture, bit-depth erosion, spectral smear, and tiny non-rhythmic digital dust suspended inside the tone. Add resonant notch movement, diffuse reverb, soft broken resampling color, and a wide unstable stereo field. It must stay sustained and ambient, not percussive or stuttering. Avoid drums, percussion, glitch beats, rhythmic stutters, clicks as rhythm, arpeggios, sequencer patterns, tremolo, delay taps, vocals, lead melody, drops, fade-outs, and silence.",
                "frozen Cm9 glitch sustain with bit-depth erosion and spectral smear",
                "granular_glitch",
                0.72f,
                76,
                6.4f,
                5.9f,
                0.0f,
            },
            {
                "ash_glass_texture",
                "Ash Glass Texture",
                "Create an original sustained synthesizer texture called Ash Glass Texture. A continuous Cm9 spectral wavetable pad and dirty glass harmonic cloud held for 64 bars, with slow phase smear, non-rhythmic wavetable drift, and dark high-frequency air. Add muted shimmer reverb, tape haze, gentle chorus, and a large cinematic stereo field. Keep it textural, ambient, and sustained without becoming a melody. Avoid drums, percussion, mallet patterns, arpeggios, sequencer patterns, rhythmic pulses, tremolo, delay taps, vocals, lead melody, drops, fade-outs, and silence.",
                "dirty Cm9 spectral glass pad with phase smear and tape haze",
                "spectral_pad",
                0.60f,
                52,
                5.9f,
                5.8f,
                0.0f,
            },
        }};
        return true;
    }
    if (config.profile == "sustained_no_decay_trials") {
        config.slots = {{
            {
                "unbroken_synth_pad",
                "Unbroken Synth Pad",
                "Create an original sustained synthesizer texture called Unbroken Synth Pad. A single Cm9 polysynth pad held continuously for 32 bars with a flat sustain level, no fade-out, and no decay after the attack. The sound is smooth, legato, beatless, and organ-like, with only very slow non-rhythmic filter color drift. Keep it as one unbroken chord tone from start to finish. Avoid drums, percussion, arpeggios, sequencer patterns, tremolo, pulsing, delay echoes, granular texture, stutters, vocals, lead melody, and silence.",
                "unbroken beatless Cm9 polysynth pad with slow non-rhythmic filter drift",
                "analog_drone",
                0.50f,
                28,
                7.6f,
                6.6f,
                0.0f,
            },
            {
                "frozen_synth_sheet",
                "Frozen Synth Sheet",
                "Create an original sustained synthesizer texture called Frozen Synth Sheet. A frozen spectral synthesizer pad holding one Cm9 harmony continuously for 32 bars, with a steady sustain level and no natural note decay. The tone is a smooth static harmonic sheet with subtle non-rhythmic spectral blur and a stable stereo field. It should remain present until the final bar without pulsing. Avoid drums, percussion, arpeggios, plucks, tremolo, rhythmic LFOs, delay taps, granular grains, vocal sounds, lead melody, fade-out, and silence.",
                "frozen beatless Cm9 spectral synth sheet with steady sustain",
                "spectral_pad",
                0.56f,
                44,
                7.6f,
                6.7f,
                0.0f,
            },
            {
                "string_machine_hold",
                "String Machine Hold",
                "Create an original sustained synthesizer texture called String Machine Hold. A warm analog string-machine style Cm9 pad held as one continuous chord for 32 bars, steady in volume, with no decay, no rhythmic modulation, and no fade-out. The movement is only slow ensemble thickness and smooth non-periodic tone color, never a pulse. Avoid drums, percussion, bass groove, arpeggios, tremolo, gated motion, delay echoes, granular texture, clicks, vocals, lead melody, endings, and silence.",
                "beatless analog string-machine Cm9 pad with steady volume",
                "spectral_pad",
                0.52f,
                36,
                7.5f,
                6.5f,
                0.0f,
            },
            {
                "sine_stack_sustain",
                "Sine Stack Sustain",
                "Create an original sustained synthesizer texture called Sine Stack Sustain. A soft stacked-sine and wavetable Cm9 drone held continuously for 32 bars at a stable level, with no envelope decay and no fade-out. The sound is pure, smooth, beatless, and sustained like an electronic organ pad, with only tiny non-rhythmic detune drift. Avoid drums, percussion, arpeggios, rhythmic filter movement, tremolo, delay taps, granular grains, clicks, stutters, vocals, lead melody, pauses, and silence.",
                "beatless stacked-sine Cm9 synth drone with stable sustain",
                "spectral_pad",
                0.50f,
                32,
                7.5f,
                6.6f,
                0.0f,
            },
        }};
        return true;
    }
    return false;
}

std::string note_name(int pitch) {
    static const std::array<const char*, 12> names = {
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    return std::string(names[pitch % 12]) + std::to_string((pitch / 12) - 1);
}

double midi_frequency(int pitch) {
    return 440.0 * std::pow(2.0, (pitch - 69) / 12.0);
}

int segment_for_time(double time_seconds, double segment_seconds) {
    return std::clamp(static_cast<int>(time_seconds / segment_seconds), 0, 3);
}

std::array<float, 4> normalize_prompt_weights(std::array<float, 4> weights) {
    float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
    if (sum <= 0.000001f || !std::isfinite(sum)) {
        return {1.0f, 0.0f, 0.0f, 0.0f};
    }
    for (float& weight : weights) weight /= sum;
    return weights;
}

std::array<float, 4> meter_sine_raw_weights(double time_seconds, const RenderConfig& config) {
    double beat_position = time_seconds * config.bpm / 60.0;
    std::array<float, 4> weights = {0.0f, 0.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < weights.size(); ++i) {
        double phase =
            2.0 * kPi * (beat_position / kMeterQuarterNotePeriods[i] + kMeterPhaseOffsets[i]);
        double lfo = 0.5 + 0.5 * std::sin(phase);
        weights[i] = static_cast<float>(0.015 + std::pow(lfo, 3.0));
    }
    return weights;
}

std::array<float, 4> macro_sine_raw_weights(double time_seconds, const RenderConfig& config) {
    double reference_seconds = std::max(0.001, config.macro_reference_seconds);
    double normalized_time = time_seconds / reference_seconds;
    double variant_phase = config.macro_phase_offset +
                           0.017 * static_cast<double>(config.batch_variant);
    std::array<float, 4> weights = {0.0f, 0.0f, 0.0f, 0.0f};
    for (size_t i = 0; i < weights.size(); ++i) {
        double slot_phase = kMacroPhaseOffsets[i] + variant_phase * static_cast<double>(i + 1);
        double phase = 2.0 * kPi * (normalized_time * kMacroCyclesPerReference[i] + slot_phase);
        double lfo = 0.5 + 0.5 * std::sin(phase);
        weights[i] = static_cast<float>(0.04 + std::pow(lfo, 1.8));
    }
    return weights;
}

std::vector<std::string> split_tab_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream stream(line);
    while (std::getline(stream, field, '\t')) {
        fields.push_back(field);
    }
    return fields;
}

std::vector<Slot> load_prompt_library(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Could not open prompt library: " + path.string());

    std::vector<Slot> slots;
    std::string line;
    bool first = true;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (first) {
            first = false;
            if (line.rfind("id\t", 0) == 0) continue;
        }
        auto fields = split_tab_line(line);
        if (fields.size() < 10) {
            throw std::runtime_error("Prompt library row has fewer than 10 fields: " + line);
        }
        Slot slot;
        slot.id = fields[0];
        slot.label = fields[1];
        slot.prompt = fields[3];
        slot.audio_recipe = fields[1] + " text prompt";
        slot.guide_kind = fields[4];
        slot.temperature = std::stof(fields[5]);
        slot.top_k = std::stoi(fields[6]);
        slot.cfg_musiccoca = std::stof(fields[7]);
        slot.cfg_notes = std::stof(fields[8]);
        slot.cfg_drums = std::stof(fields[9]);
        slots.push_back(std::move(slot));
    }
    if (slots.size() < 4) {
        throw std::runtime_error("Prompt library must contain at least 4 prompts: " + path.string());
    }
    return slots;
}

bool using_prompt_library(const RenderConfig& config) {
    return !config.prompt_library.empty();
}

int prompt_page_count(const RenderConfig& config) {
    if (!using_prompt_library(config)) return 1;
    return static_cast<int>((config.prompt_library.size() + 3) / 4);
}

int prompt_page_for_time(double time_seconds, const RenderConfig& config) {
    if (!using_prompt_library(config)) return 0;
    double page_seconds = std::max(0.001, config.prompt_page_seconds);
    int page = static_cast<int>(time_seconds / page_seconds);
    return std::clamp(page, 0, prompt_page_count(config) - 1);
}

std::array<Slot, 4> prompt_slots_for_page(const RenderConfig& config, int page) {
    if (!using_prompt_library(config)) return config.slots;
    std::array<Slot, 4> slots = config.slots;
    int base = std::max(0, page) * 4;
    for (int lane = 0; lane < 4; ++lane) {
        int index = (base + lane) % static_cast<int>(config.prompt_library.size());
        slots[lane] = config.prompt_library[index];
    }
    return slots;
}

std::array<float, 4> prompt_weights(double time_seconds, const RenderConfig& config) {
    if (config.weight_mode == "solo") {
        std::array<float, 4> weights = {0.0f, 0.0f, 0.0f, 0.0f};
        weights[std::clamp(config.solo_slot, 0, 3)] = 1.0f;
        return weights;
    }

    if (config.weight_mode == "modulated" || config.weight_mode == "polyrhythm") {
        double duration = config.duration_seconds > 0.0
            ? config.duration_seconds
            : std::max(0.001, config.segment_seconds * 4.0);
        double normalized_time = std::clamp(time_seconds / duration, 0.0, 1.0);
        constexpr std::array<double, 4> cycles = {1.0, 1.5, 2.5, 3.5};
        constexpr std::array<double, 4> phases = {0.00, 0.23, 0.47, 0.71};
        std::array<float, 4> weights = {0.0f, 0.0f, 0.0f, 0.0f};
        for (size_t i = 0; i < weights.size(); ++i) {
            double phase = 2.0 * kPi * (normalized_time * cycles[i] + phases[i]);
            double lfo = 0.5 + 0.5 * std::sin(phase);
            weights[i] = static_cast<float>(0.04 + std::pow(lfo, 1.8));
        }
        return normalize_prompt_weights(weights);
    }

    if (config.weight_mode == "page_sine") {
        double page_seconds = std::max(0.001, config.prompt_page_seconds);
        double local_time = std::fmod(std::max(0.0, time_seconds), page_seconds);
        double normalized_time = local_time / page_seconds;
        constexpr std::array<double, 4> phases = {0.00, 0.25, 0.50, 0.75};
        std::array<float, 4> weights = {0.0f, 0.0f, 0.0f, 0.0f};
        for (size_t i = 0; i < weights.size(); ++i) {
            double phase = 2.0 * kPi * (normalized_time + phases[i]);
            double lfo = 0.5 + 0.5 * std::sin(phase);
            weights[i] = static_cast<float>(0.018 + std::pow(lfo, 2.4));
        }
        return normalize_prompt_weights(weights);
    }

    if (config.weight_mode == "loop_sine") {
        double duration = config.duration_seconds > 0.0
            ? config.duration_seconds
            : std::max(0.001, config.segment_seconds * 4.0);
        double normalized_time = time_seconds / duration;
        constexpr std::array<double, 4> phases = {0.00, 0.25, 0.50, 0.75};
        std::array<float, 4> weights = {0.0f, 0.0f, 0.0f, 0.0f};
        for (size_t i = 0; i < weights.size(); ++i) {
            double phase = 2.0 * kPi * (normalized_time + phases[i]);
            // One full sine cycle over the render duration. The cubic curve
            // makes dominance obvious while a small floor keeps blends smooth.
            double lfo = 0.5 + 0.5 * std::sin(phase);
            weights[i] = static_cast<float>(0.015 + std::pow(lfo, 3.0));
        }
        return normalize_prompt_weights(weights);
    }

    if (config.weight_mode == "meter_sine") {
        return normalize_prompt_weights(meter_sine_raw_weights(time_seconds, config));
    }

    if (config.weight_mode == "meter_macro_sine") {
        auto meter = meter_sine_raw_weights(time_seconds, config);
        auto macro = macro_sine_raw_weights(time_seconds, config);
        std::array<float, 4> weights = {0.0f, 0.0f, 0.0f, 0.0f};
        for (size_t i = 0; i < weights.size(); ++i) {
            double micro = std::pow(std::max(0.0001f, meter[i]), 0.72);
            double slow = std::pow(std::max(0.0001f, macro[i]), 1.05);
            weights[i] = static_cast<float>(0.020 + 0.82 * micro * slow +
                                            0.10 * micro + 0.08 * slow);
        }
        return normalize_prompt_weights(weights);
    }

    int segment = segment_for_time(time_seconds, config.segment_seconds);
    double local = time_seconds - segment * config.segment_seconds;
    double transition_seconds = std::max(0.0, config.transition_seconds);
    int next = (segment + 1) % 4;
    std::array<float, 4> weights = {0.0f, 0.0f, 0.0f, 0.0f};
    if (transition_seconds <= 0.0 || local < config.segment_seconds - transition_seconds) {
        weights[segment] = 1.0f;
    } else {
        float amount = static_cast<float>(
            std::min(1.0, (local - (config.segment_seconds - transition_seconds)) / transition_seconds));
        weights[segment] = 1.0f - amount;
        weights[next] = amount;
    }
    return weights;
}

float blend_float(const std::array<float, 4>& weights,
                  const std::array<Slot, 4>& slots,
                  float Slot::*member) {
    float value = 0.0f;
    for (size_t i = 0; i < weights.size(); ++i) value += weights[i] * (slots[i].*member);
    return value;
}

int blend_top_k(const std::array<float, 4>& weights, const std::array<Slot, 4>& slots) {
    float value = 0.0f;
    for (size_t i = 0; i < weights.size(); ++i) value += weights[i] * slots[i].top_k;
    return static_cast<int>(std::lround(value));
}

double clamp01(double value) {
    return std::clamp(value, 0.0, 1.0);
}

double smoothstep(double value) {
    double x = clamp01(value);
    return x * x * (3.0 - 2.0 * x);
}

double render_progress(double time_seconds, const RenderConfig& config) {
    double duration = config.duration_seconds > 0.0
        ? config.duration_seconds
        : std::max(0.001, config.segment_seconds * 4.0);
    return clamp01(time_seconds / duration);
}

float control_cfg_musiccoca(const std::array<float, 4>& weights,
                            const std::array<Slot, 4>& slots,
                            double time_seconds,
                            const RenderConfig& config) {
    if (config.control_mode == "ambient_crescendo") {
        double rise = smoothstep(render_progress(time_seconds, config));
        return static_cast<float>(5.8 + 2.0 * rise);
    }
    return blend_float(weights, slots, &Slot::cfg_musiccoca);
}

float control_cfg_notes(const std::array<float, 4>& weights,
                        const std::array<Slot, 4>& slots,
                        double time_seconds,
                        const RenderConfig& config) {
    if (config.control_mode == "ambient_crescendo") {
        double rise = smoothstep(render_progress(time_seconds, config));
        return static_cast<float>(5.6 + 0.8 * rise);
    }
    return blend_float(weights, slots, &Slot::cfg_notes);
}

float control_cfg_drums(const std::array<float, 4>& weights,
                        const std::array<Slot, 4>& slots,
                        double time_seconds,
                        const RenderConfig& config) {
    (void)time_seconds;
    if (config.control_mode == "ambient_crescendo") {
        return 0.0f;
    }
    return blend_float(weights, slots, &Slot::cfg_drums);
}

float control_temperature(const std::array<float, 4>& weights,
                          const std::array<Slot, 4>& slots,
                          double time_seconds,
                          const RenderConfig& config) {
    if (config.control_mode == "ambient_crescendo") {
        double x = render_progress(time_seconds, config);
        double wobble = 0.12 * x * (1.0 - x) * std::sin(2.0 * kPi * (1.37 * x + 0.13));
        double shaped = clamp01(smoothstep(x) + wobble);
        double value = 0.6075 + (0.7205 - 0.6075) * shaped;
        return static_cast<float>(value);
    }
    return blend_float(weights, slots, &Slot::temperature);
}

int control_top_k(const std::array<float, 4>& weights,
                  const std::array<Slot, 4>& slots,
                  double time_seconds,
                  const RenderConfig& config) {
    if (config.control_mode == "ambient_crescendo") {
        double x = render_progress(time_seconds, config);
        double wobble = 0.12 * x * (1.0 - x) * std::sin(2.0 * kPi * (1.11 * x + 0.31));
        double shaped = clamp01(smoothstep(x) + wobble);
        double value = 51.0 + (103.0 - 51.0) * shaped;
        return static_cast<int>(std::lround(value));
    }
    return blend_top_k(weights, slots);
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

std::vector<float> prepare_audio_prompt_embedding_samples(const std::vector<float>& samples) {
    constexpr int kAudioPromptSampleRate = 16000;
    constexpr int kAudioPromptSeconds = 10;
    constexpr size_t kAudioPromptFrames = kAudioPromptSampleRate * kAudioPromptSeconds;
    std::vector<float> prepared(kAudioPromptFrames, 0.0f);
    if (samples.empty()) return prepared;

    const double source_per_prompt_sample =
        static_cast<double>(kSampleRate) / static_cast<double>(kAudioPromptSampleRate);
    for (size_t i = 0; i < prepared.size(); ++i) {
        size_t source_index = static_cast<size_t>(std::floor(i * source_per_prompt_sample));
        if (source_index >= samples.size()) {
            source_index %= samples.size();
        }
        prepared[i] = samples[source_index];
    }
    fade_edges(prepared, 0.04);
    return prepared;
}

void add_to(std::vector<float>& samples, int start, const std::vector<float>& tone) {
    if (start >= static_cast<int>(samples.size())) return;
    int end = std::min(static_cast<int>(samples.size()), start + static_cast<int>(tone.size()));
    for (int i = start; i < end; ++i) samples[i] += tone[i - start];
}

std::array<std::vector<int>, 4> segment_notes(const MidiData& midi, double segment_seconds) {
    std::array<std::vector<int>, 4> result;
    for (int segment = 0; segment < 4; ++segment) {
        result[segment] = active_notes_at(midi, segment * segment_seconds);
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

std::vector<float> synth_chiptune_prompt(const MidiData& midi, double segment_seconds) {
    std::vector<float> samples(static_cast<size_t>(midi.duration_seconds * kSampleRate), 0.0f);
    auto notes = segment_notes(midi, segment_seconds);
    constexpr double step_seconds = 0.125;
    int step_count = static_cast<int>(std::lround(midi.duration_seconds / step_seconds));
    for (int step = 0; step < step_count; ++step) {
        double start_seconds = step * step_seconds;
        int segment = segment_for_time(start_seconds, segment_seconds);
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

std::vector<float> synth_808_prompt(const MidiData& midi, double segment_seconds) {
    std::vector<float> samples(static_cast<size_t>(midi.duration_seconds * kSampleRate), 0.0f);
    auto notes = segment_notes(midi, segment_seconds);
    std::mt19937 rng(2405);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    for (double beat = 0.0; beat < midi.duration_seconds; beat += 0.5) {
        int segment = segment_for_time(beat, segment_seconds);
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

std::vector<float> synth_subbass_prompt(const MidiData& midi, double segment_seconds) {
    std::vector<float> samples(static_cast<size_t>(midi.duration_seconds * kSampleRate), 0.0f);
    auto notes = segment_notes(midi, segment_seconds);
    for (double beat = 0.0; beat < midi.duration_seconds; beat += 0.5) {
        int segment = segment_for_time(beat, segment_seconds);
        int root = notes[segment].empty() ? 36 : notes[segment].front() - 24;
        int start = static_cast<int>(std::lround(beat * kSampleRate));
        int length = static_cast<int>(std::lround(0.42 * kSampleRate));
        double phase = 0.0;
        for (int i = 0; i < length && start + i < static_cast<int>(samples.size()); ++i) {
            double t = i / static_cast<double>(kSampleRate);
            double f = midi_frequency(root) * (1.35 - 0.25 * std::min(1.0, t / 0.20));
            phase += 2 * kPi * f / kSampleRate;
            double env = std::min(1.0, t / 0.012) * std::exp(-5.2 * t);
            samples[start + i] += static_cast<float>(0.55 * std::sin(phase) * env);
        }
    }
    for (float& sample : samples) sample = std::tanh(sample * 2.2f);
    fade_edges(samples);
    normalize_audio(samples);
    return samples;
}

std::vector<float> synth_footwork_prompt(const MidiData& midi) {
    std::vector<float> samples(static_cast<size_t>(midi.duration_seconds * kSampleRate), 0.0f);
    std::mt19937 rng(9907);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    const std::array<double, 8> kicks = {0.0, 0.375, 0.75, 1.125, 1.5, 1.875, 2.25, 2.875};
    for (double bar = 0.0; bar < midi.duration_seconds; bar += 4.0) {
        for (double offset : kicks) {
            int start = static_cast<int>(std::lround((bar + offset) * kSampleRate));
            int length = static_cast<int>(std::lround(0.12 * kSampleRate));
            double phase = 0.0;
            for (int i = 0; i < length && start + i < static_cast<int>(samples.size()); ++i) {
                double t = i / static_cast<double>(kSampleRate);
                double f = 96.0 * (1.8 - 0.7 * std::min(1.0, t / 0.07));
                phase += 2 * kPi * f / kSampleRate;
                samples[start + i] += static_cast<float>(0.42 * std::sin(phase) * std::exp(-22.0 * t));
            }
        }
    }
    for (double snare = 0.5; snare < midi.duration_seconds; snare += 0.75) {
        int start = static_cast<int>(std::lround(snare * kSampleRate));
        int length = static_cast<int>(std::lround(0.07 * kSampleRate));
        for (int i = 0; i < length && start + i < static_cast<int>(samples.size()); ++i) {
            double t = i / static_cast<double>(kSampleRate);
            samples[start + i] += static_cast<float>(0.20 * noise(rng) * std::exp(-35.0 * t));
        }
    }
    for (double hat = 0.0; hat < midi.duration_seconds; hat += 0.0625) {
        int start = static_cast<int>(std::lround(hat * kSampleRate));
        int length = static_cast<int>(std::lround(0.025 * kSampleRate));
        for (int i = 0; i < length && start + i < static_cast<int>(samples.size()); ++i) {
            double t = i / static_cast<double>(kSampleRate);
            samples[start + i] += static_cast<float>(0.035 * noise(rng) *
                                                     std::sin(2 * kPi * 9200.0 * t) *
                                                     std::exp(-90.0 * t));
        }
    }
    for (float& sample : samples) sample = std::tanh(sample * 2.0f);
    fade_edges(samples);
    normalize_audio(samples);
    return samples;
}

std::vector<float> synth_metallic_prompt(const MidiData& midi) {
    std::vector<float> samples(static_cast<size_t>(midi.duration_seconds * kSampleRate), 0.0f);
    for (const auto& note : midi.notes) {
        int start = static_cast<int>(std::lround(note.start_seconds * kSampleRate));
        int length = static_cast<int>(std::lround(0.32 * kSampleRate));
        double f = midi_frequency(note.pitch + 24);
        for (int i = 0; i < length && start + i < static_cast<int>(samples.size()); ++i) {
            double t = i / static_cast<double>(kSampleRate);
            double env = std::min(1.0, t / 0.004) * std::exp(-9.5 * t);
            double mod = std::sin(2 * kPi * f * 2.71 * t) * 7.0 * std::exp(-7.0 * t);
            double tone = std::sin(2 * kPi * f * t + mod) +
                          0.35 * std::sin(2 * kPi * f * 3.17 * t);
            samples[start + i] += static_cast<float>(0.18 * tone * env);
        }
    }
    fade_edges(samples);
    normalize_audio(samples);
    return samples;
}

std::vector<float> synth_minimal_prompt(const MidiData& midi) {
    std::vector<float> samples(static_cast<size_t>(midi.duration_seconds * kSampleRate), 0.0f);
    for (double beat = 0.0; beat < midi.duration_seconds; beat += 0.5) {
        int start = static_cast<int>(std::lround(beat * kSampleRate));
        int length = static_cast<int>(std::lround(0.11 * kSampleRate));
        double phase = 0.0;
        for (int i = 0; i < length && start + i < static_cast<int>(samples.size()); ++i) {
            double t = i / static_cast<double>(kSampleRate);
            double f = 72.0 * (1.9 - 0.6 * std::min(1.0, t / 0.05));
            phase += 2 * kPi * f / kSampleRate;
            samples[start + i] += static_cast<float>(0.40 * std::sin(phase) * std::exp(-24.0 * t));
        }
    }
    for (const auto& note : midi.notes) {
        int start = static_cast<int>(std::lround((note.start_seconds + 0.02) * kSampleRate));
        int length = static_cast<int>(std::lround(0.16 * kSampleRate));
        double f = midi_frequency(note.pitch);
        for (int i = 0; i < length && start + i < static_cast<int>(samples.size()); ++i) {
            double t = i / static_cast<double>(kSampleRate);
            samples[start + i] += static_cast<float>(0.045 * std::sin(2 * kPi * f * t) * std::exp(-10.0 * t));
        }
    }
    fade_edges(samples);
    normalize_audio(samples);
    return samples;
}

std::vector<float> synth_analog_drone_prompt(const MidiData& midi) {
    std::vector<float> samples(static_cast<size_t>(midi.duration_seconds * kSampleRate), 0.0f);
    for (const auto& note : midi.notes) {
        int start = static_cast<int>(std::lround(note.start_seconds * kSampleRate));
        int end = std::min(static_cast<int>(samples.size()),
                           static_cast<int>(std::lround(note.end_seconds * kSampleRate)));
        double base = midi_frequency(note.pitch - 24);
        for (int i = start; i < end; ++i) {
            double t = i / static_cast<double>(kSampleRate);
            double local = (i - start) / static_cast<double>(kSampleRate);
            double dur = std::max(0.001, note.end_seconds - note.start_seconds);
            double env = std::min(1.0, local / 0.9) *
                         std::min(1.0, std::max(0.0, (dur - local) / 0.8));
            double cutoff = 0.45 + 0.55 * (0.5 + 0.5 * std::sin(2 * kPi * 0.035 * t));
            double tone = 0.58 * std::sin(2 * kPi * base * t) +
                          0.46 * std::sin(2 * kPi * base * 1.006 * t + 0.4) +
                          0.18 * std::sin(2 * kPi * base * 2.01 * t);
            samples[i] += static_cast<float>(0.10 * tone * cutoff * env);
        }
    }
    int delay = static_cast<int>(std::lround(0.43 * kSampleRate));
    for (int i = delay; i < static_cast<int>(samples.size()); ++i) {
        samples[i] += 0.28f * samples[i - delay];
    }
    for (float& sample : samples) sample = std::tanh(sample * 1.6f);
    fade_edges(samples, 0.08);
    normalize_audio(samples);
    return samples;
}

std::vector<float> synth_spectral_pad_prompt(const MidiData& midi) {
    std::vector<float> samples(static_cast<size_t>(midi.duration_seconds * kSampleRate), 0.0f);
    for (const auto& note : midi.notes) {
        int start = static_cast<int>(std::lround(note.start_seconds * kSampleRate));
        int end = std::min(static_cast<int>(samples.size()),
                           static_cast<int>(std::lround(note.end_seconds * kSampleRate)));
        double base = midi_frequency(note.pitch + 12);
        for (int i = start; i < end; ++i) {
            double t = i / static_cast<double>(kSampleRate);
            double local = (i - start) / static_cast<double>(kSampleRate);
            double dur = std::max(0.001, note.end_seconds - note.start_seconds);
            double env = std::min(1.0, local / 1.2) *
                         std::min(1.0, std::max(0.0, (dur - local) / 0.9));
            double shimmer = 0.5 + 0.5 * std::sin(2 * kPi * 0.061 * t + note.pitch);
            double tone = 0.34 * std::sin(2 * kPi * base * t) +
                          0.28 * std::sin(2 * kPi * base * 1.5 * t + 0.7 * shimmer) +
                          0.18 * std::sin(2 * kPi * base * 2.01 * t + 1.4) +
                          0.12 * std::sin(2 * kPi * base * 3.0 * t + 2.0 * shimmer);
            samples[i] += static_cast<float>(0.11 * tone * env);
        }
    }
    int delay_a = static_cast<int>(std::lround(0.17 * kSampleRate));
    int delay_b = static_cast<int>(std::lround(0.31 * kSampleRate));
    for (int i = std::max(delay_a, delay_b); i < static_cast<int>(samples.size()); ++i) {
        samples[i] += 0.18f * samples[i - delay_a] + 0.13f * samples[i - delay_b];
    }
    fade_edges(samples, 0.08);
    normalize_audio(samples);
    return samples;
}

std::vector<float> synth_cinema_noise_prompt(const MidiData& midi) {
    std::vector<float> samples(static_cast<size_t>(midi.duration_seconds * kSampleRate), 0.0f);
    std::mt19937 rng(8128);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    float low = 0.0f;
    float band = 0.0f;
    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        double t = i / static_cast<double>(kSampleRate);
        float n = noise(rng);
        float sweep = static_cast<float>(0.003 + 0.0025 * (0.5 + 0.5 * std::sin(2 * kPi * 0.027 * t)));
        low += sweep * (n - low);
        band += 0.015f * (low - band);
        samples[i] += 0.22f * band;
    }
    for (const auto& note : midi.notes) {
        int start = static_cast<int>(std::lround(note.start_seconds * kSampleRate));
        int end = std::min(static_cast<int>(samples.size()),
                           static_cast<int>(std::lround(note.end_seconds * kSampleRate)));
        double base = midi_frequency(note.pitch - 36);
        for (int i = start; i < end; ++i) {
            double t = i / static_cast<double>(kSampleRate);
            double local = (i - start) / static_cast<double>(kSampleRate);
            double env = std::min(1.0, local / 1.5);
            samples[i] += static_cast<float>(0.12 * std::sin(2 * kPi * base * t) * env);
        }
    }
    int delay = static_cast<int>(std::lround(0.61 * kSampleRate));
    for (int i = delay; i < static_cast<int>(samples.size()); ++i) {
        samples[i] += 0.24f * samples[i - delay];
    }
    for (float& sample : samples) sample = std::tanh(sample * 2.0f);
    fade_edges(samples, 0.08);
    normalize_audio(samples);
    return samples;
}

std::vector<float> synth_granular_glitch_prompt(const MidiData& midi) {
    std::vector<float> samples(static_cast<size_t>(midi.duration_seconds * kSampleRate), 0.0f);
    std::mt19937 rng(1601);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    for (double grain = 0.0; grain < midi.duration_seconds; grain += 0.083) {
        auto notes = active_notes_at(midi, grain);
        int pitch = notes.empty() ? 57 : notes[static_cast<size_t>(uni(rng) * notes.size()) % notes.size()] + 12;
        double base = midi_frequency(pitch) * (0.5 + 1.5 * uni(rng));
        int start = static_cast<int>(std::lround(grain * kSampleRate));
        int length = static_cast<int>(std::lround((0.08 + 0.16 * uni(rng)) * kSampleRate));
        for (int i = 0; i < length && start + i < static_cast<int>(samples.size()); ++i) {
            double t = i / static_cast<double>(kSampleRate);
            double env = std::sin(kPi * i / std::max(1, length));
            double smear = 0.5 + 0.5 * std::sin(2 * kPi * 0.41 * (grain + t));
            double tone = std::sin(2 * kPi * base * t + 4.0 * smear) +
                          0.22 * noise(rng);
            samples[start + i] += static_cast<float>(0.07 * tone * env);
        }
    }
    int delay = static_cast<int>(std::lround(0.12 * kSampleRate));
    for (int i = delay; i < static_cast<int>(samples.size()); ++i) {
        samples[i] += 0.32f * samples[i - delay];
    }
    for (float& sample : samples) {
        sample = std::tanh(sample * 2.4f);
        sample = std::round(sample * 96.0f) / 96.0f;
    }
    fade_edges(samples, 0.08);
    normalize_audio(samples);
    return samples;
}

std::vector<float> synth_for_guide_kind(const MidiData& midi,
                                        const std::string& guide_kind,
                                        double segment_seconds) {
    if (guide_kind == "piano") return synth_piano_prompt(midi);
    if (guide_kind == "chiptune") return synth_chiptune_prompt(midi, segment_seconds);
    if (guide_kind == "drums808") return synth_808_prompt(midi, segment_seconds);
    if (guide_kind == "pad") return synth_pad_prompt(midi);
    if (guide_kind == "subbass") return synth_subbass_prompt(midi, segment_seconds);
    if (guide_kind == "footwork") return synth_footwork_prompt(midi);
    if (guide_kind == "metallic") return synth_metallic_prompt(midi);
    if (guide_kind == "minimal") return synth_minimal_prompt(midi);
    if (guide_kind == "analog_drone") return synth_analog_drone_prompt(midi);
    if (guide_kind == "spectral_pad") return synth_spectral_pad_prompt(midi);
    if (guide_kind == "cinema_noise") return synth_cinema_noise_prompt(midi);
    if (guide_kind == "granular_glitch") return synth_granular_glitch_prompt(midi);
    return synth_pad_prompt(midi);
}

std::array<std::vector<float>, 4> build_audio_prompts(const MidiData& midi,
                                                      const std::array<Slot, 4>& slots,
                                                      double segment_seconds) {
    return {synth_for_guide_kind(midi, slots[0].guide_kind, segment_seconds),
            synth_for_guide_kind(midi, slots[1].guide_kind, segment_seconds),
            synth_for_guide_kind(midi, slots[2].guide_kind, segment_seconds),
            synth_for_guide_kind(midi, slots[3].guide_kind, segment_seconds)};
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

void limit_peak(std::vector<float>& interleaved) {
    float peak = 0.0f;
    for (float v : interleaved) peak = std::max(peak, std::abs(v));
    float target_peak = std::pow(10.0f, -1.0f / 20.0f);
    if (peak > target_peak) {
        float gain = target_peak / peak;
        for (float& v : interleaved) v *= gain;
    }
}

void match_segment_rms(std::vector<float>& interleaved,
                       double duration_seconds,
                       double segment_seconds,
                       float target_rms) {
    int total_frames = static_cast<int>(interleaved.size() / 2);
    for (int segment = 0; segment < 4; ++segment) {
        int start = static_cast<int>(std::lround(segment * segment_seconds * kSampleRate));
        int end = std::min(total_frames, static_cast<int>(std::lround((segment + 1) * segment_seconds * kSampleRate)));
        if (segment * segment_seconds >= duration_seconds) break;
        float rms = rms_segment(interleaved, start, end);
        float gain = rms > 0.000001f ? std::min(8.0f, target_rms / rms) : 1.0f;
        for (int frame = start; frame < end; ++frame) {
            interleaved[frame * 2] *= gain;
            interleaved[frame * 2 + 1] *= gain;
        }
    }
    limit_peak(interleaved);
}

void stabilize_window_rms(std::vector<float>& interleaved,
                          double window_seconds,
                          float min_window_rms,
                          float max_window_gain) {
    int total_frames = static_cast<int>(interleaved.size() / 2);
    int window_frames = std::max(1, static_cast<int>(std::lround(window_seconds * kSampleRate)));
    for (int start = 0; start < total_frames; start += window_frames) {
        int end = std::min(total_frames, start + window_frames);
        float rms = rms_segment(interleaved, start, end);
        if (rms <= 0.00000001f || rms >= min_window_rms) continue;
        float gain = std::min(max_window_gain, min_window_rms / rms);
        for (int frame = start; frame < end; ++frame) {
            interleaved[frame * 2] *= gain;
            interleaved[frame * 2 + 1] *= gain;
        }
    }
    limit_peak(interleaved);
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
    out << "  \"profile\": \"" << json_escape(config.profile) << "\",\n";
    out << "  \"weight_mode\": \"" << json_escape(config.weight_mode) << "\",\n";
    out << "  \"control_mode\": \"" << json_escape(config.control_mode) << "\",\n";
    out << "  \"solo_slot\": " << config.solo_slot << ",\n";
    out << "  \"weights_output\": \"" << json_escape(config.weights_path.string()) << "\",\n";
    out << "  \"prompt_library\": \"" << json_escape(config.prompt_library_path.string()) << "\",\n";
    out << "  \"prompt_library_size\": " << config.prompt_library.size() << ",\n";
    out << "  \"prompt_page_seconds\": " << config.prompt_page_seconds << ",\n";
    out << "  \"batch_variant\": " << config.batch_variant << ",\n";
    out << "  \"macro_phase_offset\": " << config.macro_phase_offset << ",\n";
    out << "  \"macro_reference_seconds\": " << config.macro_reference_seconds << ",\n";
    out << "  \"midi_mode\": \"" << json_escape(config.midi_mode) << "\",\n";
    out << "  \"midi_refresh_seconds\": " << config.midi_refresh_seconds << ",\n";
    out << "  \"bpm\": " << config.bpm << ",\n";
    out << "  \"model\": \"" << json_escape(config.model_name) << "\",\n";
    out << "  \"model_path\": \"" << json_escape(model_path) << "\",\n";
    out << "  \"embedding_source\": \"" << (config.use_audio_prompts ? "audio" : "text") << "\",\n";
    out << "  \"duration_seconds\": " << (static_cast<double>(total_frames) / kSampleRate) << ",\n";
    out << "  \"expected_duration_seconds\": " << midi.duration_seconds << ",\n";
    out << "  \"sample_rate\": " << kSampleRate << ",\n";
    out << "  \"channels\": 2,\n";
    out << "  \"frames_25hz\": " << static_cast<int>(std::lround(midi.duration_seconds * kFps)) << ",\n";
    out << "  \"chunk_samples\": " << kFrameSamples << ",\n";
    out << "  \"segment_seconds\": " << config.segment_seconds << ",\n";
    out << "  \"transition_seconds\": " << config.transition_seconds << ",\n";
    out << "  \"stabilize_window_rms\": " << (config.stabilize_window_rms ? "true" : "false") << ",\n";
    out << "  \"min_window_rms\": " << config.min_window_rms << ",\n";
    out << "  \"max_window_gain\": " << config.max_window_gain << ",\n";
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
    out << "    \"secondary\": \"cfg_musiccoca / cfg_notes / cfg_drums, optionally shaped by control_mode\",\n";
    out << "    \"expression\": \"temperature, optionally shaped by control_mode\",\n";
    out << "    \"exploration\": \"top_k, optionally shaped by control_mode\",\n";
    out << "    \"stability\": \"fixed 25 Hz frames / 1920 sample chunks\",\n";
    out << "    \"midi\": \"MLXEngine::set_note_on/off -> MidiNoteTracker -> generate_frame\"\n";
    out << "  },\n";
    out << "  \"segments\": [\n";
    for (int segment = 0; segment < 4; ++segment) {
        int start = static_cast<int>(std::lround(segment * config.segment_seconds * kSampleRate));
        int end = std::min(total_frames, static_cast<int>(std::lround((segment + 1) * config.segment_seconds * kSampleRate)));
        SegmentMetrics metrics = metrics_segment(interleaved, start, end);
        auto notes = active_notes_at(midi, segment * config.segment_seconds);
        double segment_time = segment * config.segment_seconds;
        auto weights = prompt_weights(segment_time, config);
        out << "    {\n";
        out << "      \"segment\": " << (segment + 1) << ",\n";
        out << "      \"slot_id\": \"" << config.slots[segment].id << "\",\n";
        out << "      \"slot_label\": \"" << config.slots[segment].label << "\",\n";
        out << "      \"prompt\": \"" << json_escape(config.slots[segment].prompt) << "\",\n";
        out << "      \"guide_kind\": \"" << json_escape(config.slots[segment].guide_kind) << "\",\n";
        out << "      \"weight\": " << weights[segment] << ",\n";
        out << "      \"cfg_musiccoca\": " << control_cfg_musiccoca(weights, config.slots, segment_time, config) << ",\n";
        out << "      \"cfg_notes\": " << control_cfg_notes(weights, config.slots, segment_time, config) << ",\n";
        out << "      \"cfg_drums\": " << control_cfg_drums(weights, config.slots, segment_time, config) << ",\n";
        out << "      \"temperature\": " << control_temperature(weights, config.slots, segment_time, config) << ",\n";
        out << "      \"top_k\": " << control_top_k(weights, config.slots, segment_time, config) << ",\n";
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

bool write_weight_frames(const std::filesystem::path& path,
                         const RenderConfig& config,
                         int frame_count) {
    if (path.empty()) return true;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out) return false;

    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"schema\": \"mrt-cpp-prompt-weight-frames-v1\",\n";
    out << "  \"profile\": \"" << json_escape(config.profile) << "\",\n";
    out << "  \"weight_mode\": \"" << json_escape(config.weight_mode) << "\",\n";
    out << "  \"control_mode\": \"" << json_escape(config.control_mode) << "\",\n";
    out << "  \"solo_slot\": " << config.solo_slot << ",\n";
    out << "  \"bpm\": " << config.bpm << ",\n";
    out << "  \"duration_seconds\": " << config.duration_seconds << ",\n";
    out << "  \"frame_rate\": " << kWeightFrameRate << ",\n";
    out << "  \"prompt_library\": \"" << json_escape(config.prompt_library_path.string()) << "\",\n";
    out << "  \"prompt_library_size\": " << config.prompt_library.size() << ",\n";
    out << "  \"prompt_page_seconds\": " << config.prompt_page_seconds << ",\n";
    out << "  \"batch_variant\": " << config.batch_variant << ",\n";
    out << "  \"macro_phase_offset\": " << config.macro_phase_offset << ",\n";
    out << "  \"macro_reference_seconds\": " << config.macro_reference_seconds << ",\n";
    out << "  \"midi_mode\": \"" << json_escape(config.midi_mode) << "\",\n";
    out << "  \"midi_refresh_seconds\": " << config.midi_refresh_seconds << ",\n";
    if (config.weight_mode == "meter_sine" || config.weight_mode == "meter_macro_sine") {
        out << "  \"meter_sine\": [\n";
        out << "    {\"meter\": \"4/4\", \"quarter_note_period\": " << kMeterQuarterNotePeriods[0]
            << ", \"phase_offset\": " << kMeterPhaseOffsets[0] << "},\n";
        out << "    {\"meter\": \"3/4\", \"quarter_note_period\": " << kMeterQuarterNotePeriods[1]
            << ", \"phase_offset\": " << kMeterPhaseOffsets[1] << "},\n";
        out << "    {\"meter\": \"6/8\", \"quarter_note_period\": " << kMeterQuarterNotePeriods[2]
            << ", \"phase_offset\": " << kMeterPhaseOffsets[2] << "},\n";
        out << "    {\"meter\": \"4/4\", \"quarter_note_period\": " << kMeterQuarterNotePeriods[3]
            << ", \"phase_offset\": " << kMeterPhaseOffsets[3] << "}\n";
        out << "  ],\n";
    }
    if (config.weight_mode == "meter_macro_sine") {
        out << "  \"macro_sine\": [\n";
        for (int i = 0; i < 4; ++i) {
            out << "    {\"slot\": " << i
                << ", \"cycles_per_reference\": " << kMacroCyclesPerReference[i]
                << ", \"base_phase_offset\": " << kMacroPhaseOffsets[i]
                << ", \"reference_seconds\": " << config.macro_reference_seconds
                << "}" << (i == 3 ? "\n" : ",\n");
        }
        out << "  ],\n";
    }
    out << "  \"slots\": [\n";
    for (int i = 0; i < 4; ++i) {
        out << "    {\"index\": " << i << ", \"id\": \"" << json_escape(config.slots[i].id)
            << "\", \"label\": \"" << json_escape(config.slots[i].label) << "\"}"
            << (i == 3 ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"prompt_pages\": [\n";
    int pages = prompt_page_count(config);
    for (int page = 0; page < pages; ++page) {
        auto page_slots = prompt_slots_for_page(config, page);
        out << "    {\"page\": " << page
            << ", \"start_seconds\": " << (page * config.prompt_page_seconds)
            << ", \"end_seconds\": " << ((page + 1) * config.prompt_page_seconds)
            << ", \"prompt_indices\": [";
        for (int lane = 0; lane < 4; ++lane) {
            if (lane) out << ", ";
            out << ((page * 4 + lane) % std::max(1, static_cast<int>(config.prompt_library.size())));
        }
        out << "], \"ids\": [";
        for (int lane = 0; lane < 4; ++lane) {
            if (lane) out << ", ";
            out << "\"" << json_escape(page_slots[lane].id) << "\"";
        }
        out << "], \"labels\": [";
        for (int lane = 0; lane < 4; ++lane) {
            if (lane) out << ", ";
            out << "\"" << json_escape(page_slots[lane].label) << "\"";
        }
        out << "]}" << (page + 1 == pages ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"frames\": [\n";
    for (int frame = 0; frame < frame_count; ++frame) {
        double time_seconds = frame * kWeightFrameSeconds;
        int page = prompt_page_for_time(time_seconds, config);
        auto active_slots = prompt_slots_for_page(config, page);
        auto weights = prompt_weights(time_seconds, config);
        out << "    {\"frame\": " << frame
            << ", \"time_seconds\": " << time_seconds
            << ", \"prompt_page\": " << page
            << ", \"active_slot_ids\": [\"" << json_escape(active_slots[0].id) << "\", \""
            << json_escape(active_slots[1].id) << "\", \"" << json_escape(active_slots[2].id)
            << "\", \"" << json_escape(active_slots[3].id) << "\"]"
            << ", \"active_slot_labels\": [\"" << json_escape(active_slots[0].label) << "\", \""
            << json_escape(active_slots[1].label) << "\", \"" << json_escape(active_slots[2].label)
            << "\", \"" << json_escape(active_slots[3].label) << "\"]"
            << ", \"weights\": [" << weights[0] << ", " << weights[1] << ", "
            << weights[2] << ", " << weights[3] << "]"
            << ", \"cfg_musiccoca\": " << control_cfg_musiccoca(weights, active_slots, time_seconds, config)
            << ", \"cfg_notes\": " << control_cfg_notes(weights, active_slots, time_seconds, config)
            << ", \"cfg_drums\": " << control_cfg_drums(weights, active_slots, time_seconds, config)
            << ", \"temperature\": " << control_temperature(weights, active_slots, time_seconds, config)
            << ", \"top_k\": " << control_top_k(weights, active_slots, time_seconds, config)
            << "}" << (frame + 1 == frame_count ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
    return out.good();
}

bool reblend_prompt_weights(MLXEngine& engine,
                            const std::array<float, 4>& weights,
                            bool use_audio_prompts) {
    if (use_audio_prompts) {
        std::array<float, 6> audio_weights = {
            weights[0], weights[1], weights[2], weights[3], 0.0f, 0.0f};
        return engine.reblend_musiccoca_tokens(audio_weights.data(),
                                               static_cast<int>(audio_weights.size()));
    }
    return engine.reblend_musiccoca_tokens(weights.data(), static_cast<int>(weights.size()));
}

void wait_for_prompts(MLXEngine& engine, int slot_count) {
    constexpr int kMaxWaitIterations = 30000;
    for (int iteration = 0; iteration < kMaxWaitIterations; ++iteration) {
        bool busy = engine.get_text_encoder_status() == 1 ||
                    engine.get_quantizer_status() == 1;
        bool all_ready = true;
        for (int i = 0; i < slot_count; ++i) {
            int status = engine.get_prompt_status(i);
            if (status == 3) {
                std::fprintf(stderr, "Prompt slot %d failed to encode\n", i + 1);
                return;
            }
            if (status != 2) {
                all_ready = false;
            }
        }
        if (!busy && all_ready) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::fprintf(stderr, "Timed out waiting for prompt encoding\n");
}

void set_text_prompt_slots(MLXEngine& engine, const std::array<Slot, 4>& slots) {
    std::vector<std::string> prompts;
    std::vector<float> initial_weights;
    prompts.reserve(slots.size());
    initial_weights.reserve(slots.size());
    for (int i = 0; i < 4; ++i) {
        prompts.push_back(slots[i].prompt);
        initial_weights.push_back(i == 0 ? 1.0f : 0.0f);
    }
    engine.set_text_prompts(prompts, initial_weights);
    wait_for_prompts(engine, static_cast<int>(slots.size()));
}

void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --midi PATH              MIDI file (default: assets/Am-Minor Prog 01 (i-VI-v-iv).mid)\n"
        "  --output PATH            Output WAV path\n"
        "  --report PATH            Output JSON report path\n"
        "  --model NAME             Model folder under Magenta models (default: mrt2_small)\n"
        "  --resources PATH         Resource dir containing musiccoca/\n"
        "  --profile NAME           clear_extreme, microcinematic_footwork, negative_space_club,\n"
        "                           glass_trap_pressure, metallic_ambient_bounce,\n"
        "                           sustained_synth_textures, sustained_no_decay_trials,\n"
        "                           or dirty_cinematic_ambient_cm9\n"
        "  --weight-mode NAME       sequential, solo, modulated, polyrhythm, loop_sine,\n"
        "                           meter_sine, meter_macro_sine, or page_sine\n"
        "  --control-mode NAME      slot_blend or ambient_crescendo\n"
        "  --solo-slot INDEX        1-4 prompt slot used when --weight-mode solo\n"
        "  --weights-output PATH    Output frame-level prompt weight JSON\n"
        "  --prompt-library PATH    TSV library of prompts to page through four at a time\n"
        "  --prompt-page-seconds N  Seconds per four-prompt library page (default: 4.0)\n"
        "  --macro-reference-seconds N  Reference duration for meter_macro_sine macro LFOs\n"
        "  --macro-phase-offset N   Extra phase offset for meter_macro_sine macro LFOs\n"
        "  --batch-variant INDEX    Variant index folded into meter_macro_sine macro phases\n"
        "  --midi-mode NAME         scheduled or initial_latch (default: scheduled)\n"
        "  --midi-refresh-seconds N Re-send active MIDI notes every N seconds during render\n"
        "  --duration SECONDS       Repeat/clip the MIDI progression to this duration\n"
        "  --transition SECONDS     Prompt crossfade duration at segment boundaries\n"
        "  --text-prompts           Use text prompts instead of MIDI-derived audio prompt embeddings\n"
        "  --no-match-rms           Do not RMS-match the four prompt sections\n"
        "  --target-rms VALUE       Target RMS when matching sections (default: 0.045)\n"
        "  --stabilize-window-rms  Raise very quiet 1s windows after segment RMS matching\n"
        "  --min-window-rms VALUE   Minimum RMS for stabilized windows (default: 0.002)\n"
        "  --max-window-gain VALUE  Maximum gain for stabilized windows (default: 256)\n",
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
        } else if (arg == "--profile") {
            config.profile = need_value("--profile");
        } else if (arg == "--weight-mode") {
            config.weight_mode = need_value("--weight-mode");
        } else if (arg == "--control-mode") {
            config.control_mode = need_value("--control-mode");
        } else if (arg == "--solo-slot") {
            config.solo_slot = std::stoi(need_value("--solo-slot")) - 1;
        } else if (arg == "--weights-output") {
            config.weights_path = need_value("--weights-output");
        } else if (arg == "--prompt-library") {
            config.prompt_library_path = need_value("--prompt-library");
        } else if (arg == "--prompt-page-seconds") {
            config.prompt_page_seconds = std::stod(need_value("--prompt-page-seconds"));
        } else if (arg == "--macro-reference-seconds") {
            config.macro_reference_seconds = std::stod(need_value("--macro-reference-seconds"));
        } else if (arg == "--macro-phase-offset") {
            config.macro_phase_offset = std::stod(need_value("--macro-phase-offset"));
        } else if (arg == "--batch-variant") {
            config.batch_variant = std::stoi(need_value("--batch-variant"));
        } else if (arg == "--midi-mode") {
            config.midi_mode = need_value("--midi-mode");
        } else if (arg == "--midi-refresh-seconds") {
            config.midi_refresh_seconds = std::stod(need_value("--midi-refresh-seconds"));
        } else if (arg == "--duration") {
            config.duration_seconds = std::stod(need_value("--duration"));
        } else if (arg == "--transition") {
            config.transition_seconds = std::stod(need_value("--transition"));
        } else if (arg == "--text-prompts") {
            config.use_audio_prompts = false;
        } else if (arg == "--no-match-rms") {
            config.match_segment_rms = false;
        } else if (arg == "--target-rms") {
            config.target_rms = std::stof(need_value("--target-rms"));
        } else if (arg == "--stabilize-window-rms") {
            config.stabilize_window_rms = true;
        } else if (arg == "--min-window-rms") {
            config.min_window_rms = std::stof(need_value("--min-window-rms"));
        } else if (arg == "--max-window-gain") {
            config.max_window_gain = std::stof(need_value("--max-window-gain"));
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
    if (!apply_profile(config)) {
        std::fprintf(stderr, "Unknown profile: %s\n", config.profile.c_str());
        return 1;
    }
    if (config.weight_mode != "sequential" &&
        config.weight_mode != "solo" &&
        config.weight_mode != "modulated" &&
        config.weight_mode != "polyrhythm" &&
        config.weight_mode != "loop_sine" &&
        config.weight_mode != "meter_sine" &&
        config.weight_mode != "meter_macro_sine" &&
        config.weight_mode != "page_sine") {
        std::fprintf(stderr, "Unknown weight mode: %s\n", config.weight_mode.c_str());
        return 1;
    }
    if (config.macro_reference_seconds <= 0.0) {
        std::fprintf(stderr, "--macro-reference-seconds must be positive\n");
        return 1;
    }
    if (config.min_window_rms <= 0.0f || config.max_window_gain <= 0.0f) {
        std::fprintf(stderr, "--min-window-rms and --max-window-gain must be positive\n");
        return 1;
    }
    if (config.midi_refresh_seconds < 0.0) {
        std::fprintf(stderr, "--midi-refresh-seconds must be non-negative\n");
        return 1;
    }
    if (config.midi_mode != "scheduled" &&
        config.midi_mode != "initial_latch") {
        std::fprintf(stderr, "Unknown MIDI mode: %s\n", config.midi_mode.c_str());
        return 1;
    }
    if (config.control_mode != "slot_blend" &&
        config.control_mode != "ambient_crescendo") {
        std::fprintf(stderr, "Unknown control mode: %s\n", config.control_mode.c_str());
        return 1;
    }
    config.solo_slot = std::clamp(config.solo_slot, 0, 3);
    if (!config.prompt_library_path.empty()) {
        config.prompt_library = load_prompt_library(config.prompt_library_path);
        config.slots = prompt_slots_for_page(config, 0);
        config.use_audio_prompts = false;
        if (config.prompt_page_seconds <= 0.0) {
            std::fprintf(stderr, "--prompt-page-seconds must be positive\n");
            return 1;
        }
    }

    try {
        MidiData midi = parse_midi(config.midi_path);
        config.bpm = midi.bpm;
        double render_duration = config.duration_seconds > 0.0
            ? config.duration_seconds
            : midi.duration_seconds;
        midi = repeat_midi_to_duration(midi, render_duration);
        config.duration_seconds = midi.duration_seconds;
        config.segment_seconds = midi.duration_seconds / 4.0;
        int frame_count = static_cast<int>(std::lround(midi.duration_seconds * kFps));
        int weight_frame_count =
            static_cast<int>(std::lround(midi.duration_seconds * kWeightFrameRate));
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
            auto prompt_audio = build_audio_prompts(midi, config.slots, config.segment_seconds);
            std::filesystem::create_directories(config.audio_prompt_dir);
            for (int i = 0; i < 4; ++i) {
                auto audio_path = config.audio_prompt_dir /
                    (std::to_string(i + 1) + "_" + config.slots[i].id + ".wav");
                write_wav(audio_path, to_stereo_interleaved(prompt_audio[i]), kSampleRate, 2);
                auto embedding_audio = prepare_audio_prompt_embedding_samples(prompt_audio[i]);
                engine.set_audio_prompt_samples(i, audio_path.filename().string(),
                                                embedding_audio.data(), embedding_audio.size());
            }
            wait_for_prompts(engine, static_cast<int>(config.slots.size()));
            std::array<float, 4> initial_weights = {1.0f, 0.0f, 0.0f, 0.0f};
            if (!reblend_prompt_weights(engine, initial_weights, config.use_audio_prompts)) {
                std::fprintf(stderr, "Initial audio prompt reblend failed\n");
                return 1;
            }
        } else {
            set_text_prompt_slots(engine, config.slots);
        }

        engine.set_onset_mode(1);
        engine.set_unmask_width(127);
        engine.set_drumless(false);
        engine.reset_state();

        // Match the official live-MIDI path during preroll too, so the model
        // is already conditioned by the first held chord before audible output.
        auto initial_notes = active_notes_at(midi, 0.0);
        for (int pitch : initial_notes) {
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
        int current_prompt_page = 0;
        std::array<Slot, 4> current_slots = config.slots;
        double next_midi_refresh_seconds = config.midi_refresh_seconds > 0.0
            ? config.midi_refresh_seconds
            : std::numeric_limits<double>::infinity();
        for (int frame = 0; frame < frame_count; ++frame) {
            double time_seconds = frame * kFrameSeconds;
            if (config.midi_mode == "scheduled") {
                while (next_event < midi.events.size() &&
                       midi.events[next_event].time_seconds <= time_seconds + 0.000001) {
                    if (midi.events[next_event].on) {
                        engine.set_note_on(midi.events[next_event].pitch);
                    } else {
                        engine.set_note_off(midi.events[next_event].pitch);
                    }
                    ++next_event;
                }
            }

            if (using_prompt_library(config)) {
                int next_prompt_page = prompt_page_for_time(time_seconds, config);
                if (next_prompt_page != current_prompt_page) {
                    current_prompt_page = next_prompt_page;
                    current_slots = prompt_slots_for_page(config, current_prompt_page);
                    set_text_prompt_slots(engine, current_slots);
                }
            }

            if (config.midi_mode == "scheduled" &&
                config.midi_refresh_seconds > 0.0 &&
                time_seconds + 0.000001 >= next_midi_refresh_seconds) {
                auto refreshed_notes = active_notes_at(midi, time_seconds);
                for (int pitch : refreshed_notes) {
                    engine.set_note_on(pitch);
                }
                while (next_midi_refresh_seconds <= time_seconds + 0.000001) {
                    next_midi_refresh_seconds += config.midi_refresh_seconds;
                }
            }

            auto weights4 = prompt_weights(time_seconds, config);
            if (!reblend_prompt_weights(engine, weights4, config.use_audio_prompts)) {
                std::fprintf(stderr, "prompt reblend failed at frame %d\n", frame);
                return 1;
            }
            engine.set_cfg_musiccoca(control_cfg_musiccoca(weights4, current_slots, time_seconds, config));
            engine.set_cfg_notes(control_cfg_notes(weights4, current_slots, time_seconds, config));
            engine.set_cfg_drums(control_cfg_drums(weights4, current_slots, time_seconds, config));
            engine.set_temperature(control_temperature(weights4, current_slots, time_seconds, config));
            engine.set_top_k(control_top_k(weights4, current_slots, time_seconds, config));

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
        if (config.midi_mode == "scheduled") {
            while (next_event < midi.events.size()) {
                if (midi.events[next_event].on) {
                    engine.set_note_on(midi.events[next_event].pitch);
                } else {
                    engine.set_note_off(midi.events[next_event].pitch);
                }
                ++next_event;
            }
        }

        if (config.match_segment_rms) {
            match_segment_rms(interleaved, midi.duration_seconds, config.segment_seconds, config.target_rms);
        }
        if (config.stabilize_window_rms) {
            stabilize_window_rms(interleaved, 1.0, config.min_window_rms, config.max_window_gain);
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
        if (!write_weight_frames(config.weights_path, config, weight_frame_count)) {
            std::fprintf(stderr, "Failed to write %s\n", config.weights_path.string().c_str());
            return 1;
        }
        std::printf("Wrote %s\n", config.output_path.string().c_str());
        std::printf("Wrote %s\n", config.report_path.string().c_str());
        if (!config.weights_path.empty()) {
            std::printf("Wrote %s\n", config.weights_path.string().c_str());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
