#pragma once

#include <array>
#include <atomic>
#include <cstdint>

namespace franky {

/**
 * @brief Wait-free single-writer/single-reader cell holding the latest published value.
 *
 * Three slots: one owned by the writer, one owned by the reader, and one in the atomic
 * "mailbox". set() publishes by swapping the writer slot into the mailbox; get() claims
 * pending data by swapping the reader slot with the mailbox. A slot is only ever accessed
 * by the thread that currently owns it, and ownership transfers through acquire/release
 * exchanges, so reads can never tear and neither side ever loops or blocks. This makes it
 * suitable for publishing values from a non-realtime thread to a 1 kHz RT control loop.
 *
 * Validity is tracked separately: hasValue() turns true on the first set() and false on
 * clear(); get() keeps returning the last published (or initial) value regardless.
 *
 * Thread safety: at most one thread may call set() or clear() at a time, and at most one
 * thread may call get() at a time. hasValue() is safe from either thread.
 */
template <typename T>
class RealtimeValue {
 public:
  RealtimeValue() = default;

  explicit RealtimeValue(const T &initial) : buffers_{initial, initial, initial} {}

  //! Publish a new value. Writer thread only.
  void set(const T &value) {
    buffers_[write_index_] = value;
    write_index_ = mailbox_.exchange(write_index_ | kNewData, std::memory_order_acq_rel) & kIndexMask;
    valid_.store(true, std::memory_order_release);
  }

  //! Mark the cell as holding no externally supplied value. Writer thread only.
  void clear() { valid_.store(false, std::memory_order_release); }

  //! Whether a value has been published and not cleared.
  [[nodiscard]] bool hasValue() const { return valid_.load(std::memory_order_acquire); }

  /**
   * @brief The latest published value. Reader thread only.
   *
   * The returned reference stays valid and unchanged until the next get() call: the
   * writer can never touch the reader-owned slot it points into.
   */
  [[nodiscard]] const T &get() const {
    if (mailbox_.load(std::memory_order_relaxed) & kNewData) {
      read_index_ = mailbox_.exchange(read_index_, std::memory_order_acq_rel) & kIndexMask;
    }
    return buffers_[read_index_];
  }

 private:
  static constexpr uint8_t kIndexMask = 0x03;
  static constexpr uint8_t kNewData = 0x04;

  // Invariant: {write_index_, read_index_, mailbox_ & kIndexMask} is a permutation of {0, 1, 2}.
  std::array<T, 3> buffers_{};
  mutable std::atomic<uint8_t> mailbox_{0};  // mutable: the const reader claims pending data via exchange
  uint8_t write_index_{1};
  mutable uint8_t read_index_{2};  // reader-thread state; only mutated from get()
  std::atomic<bool> valid_{false};
};

}  // namespace franky
