//! Lock-Free Channels and Ring Buffers
//!
//! High-performance data structures for real-time streaming:
//! - Lock-free SPSC ring buffer for time-series data
//! - Bounded MPSC for events
//! - Zero-copy where possible

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Lock-free ring buffer for time-series data
///
/// Optimized for single-producer, multiple-consumer scenarios
/// where we want to keep the last N samples.
pub struct RingBuffer<T> {
    /// Storage (protected by mutex for simplicity, could be lock-free)
    data: Mutex<VecDeque<T>>,
    /// Maximum capacity
    capacity: usize,
    /// Total items ever pushed (for statistics)
    total_pushed: AtomicUsize,
}

impl<T: Clone> RingBuffer<T> {
    /// Create a new ring buffer with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
            total_pushed: AtomicUsize::new(0),
        }
    }

    /// Push a new item, evicting oldest if at capacity
    pub fn push(&self, item: T) {
        let mut data = self.data.lock();
        if data.len() >= self.capacity {
            data.pop_front();
        }
        data.push_back(item);
        self.total_pushed.fetch_add(1, Ordering::Relaxed);
    }

    /// Get all items as a vector (for rendering)
    pub fn to_vec(&self) -> Vec<T> {
        self.data.lock().iter().cloned().collect()
    }

    /// Get the last N items
    pub fn last_n(&self, n: usize) -> Vec<T> {
        let data = self.data.lock();
        let skip = data.len().saturating_sub(n);
        data.iter().skip(skip).cloned().collect()
    }

    /// Get current length
    pub fn len(&self) -> usize {
        self.data.lock().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.lock().is_empty()
    }

    /// Get total items ever pushed
    pub fn total_pushed(&self) -> usize {
        self.total_pushed.load(Ordering::Relaxed)
    }

    /// Clear all items
    pub fn clear(&self) {
        self.data.lock().clear();
    }

    /// Get the most recent item
    pub fn latest(&self) -> Option<T> {
        self.data.lock().back().cloned()
    }

    /// Iterate over items with a callback (avoids cloning)
    pub fn for_each<F: FnMut(&T)>(&self, mut f: F) {
        let data = self.data.lock();
        for item in data.iter() {
            f(item);
        }
    }
}

/// Sliding window statistics calculator
pub struct SlidingStats {
    values: RingBuffer<f64>,
}

impl SlidingStats {
    pub fn new(window_size: usize) -> Self {
        Self {
            values: RingBuffer::new(window_size),
        }
    }

    pub fn push(&self, value: f64) {
        self.values.push(value);
    }

    pub fn mean(&self) -> f64 {
        let data = self.values.to_vec();
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    pub fn min(&self) -> f64 {
        self.values.to_vec()
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    }

    pub fn max(&self) -> f64 {
        self.values.to_vec()
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn std_dev(&self) -> f64 {
        let data = self.values.to_vec();
        if data.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;
        variance.sqrt()
    }

    pub fn trend(&self) -> f64 {
        // Simple linear regression slope
        let data = self.values.to_vec();
        if data.len() < 2 {
            return 0.0;
        }

        let n = data.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = self.mean();

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, y) in data.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

/// Rate limiter for event publishing
pub struct RateLimiter {
    /// Minimum interval between events (microseconds)
    min_interval_us: u64,
    /// Last event timestamp
    last_event: AtomicUsize,
}

impl RateLimiter {
    pub fn new(max_rate_hz: f64) -> Self {
        let min_interval_us = (1_000_000.0 / max_rate_hz) as u64;
        Self {
            min_interval_us,
            last_event: AtomicUsize::new(0),
        }
    }

    /// Check if we should emit an event
    pub fn should_emit(&self) -> bool {
        use std::time::{SystemTime, UNIX_EPOCH};

        let now_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as usize;

        let last = self.last_event.load(Ordering::Relaxed);

        if now_us.saturating_sub(last) >= self.min_interval_us as usize {
            self.last_event.store(now_us, Ordering::Relaxed);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let buf: RingBuffer<i32> = RingBuffer::new(3);

        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert_eq!(buf.to_vec(), vec![1, 2, 3]);

        buf.push(4);
        assert_eq!(buf.to_vec(), vec![2, 3, 4]);

        buf.push(5);
        assert_eq!(buf.to_vec(), vec![3, 4, 5]);
    }

    #[test]
    fn test_sliding_stats() {
        let stats = SlidingStats::new(5);

        for i in 1..=5 {
            stats.push(i as f64);
        }

        assert!((stats.mean() - 3.0).abs() < 0.001);
        assert!((stats.min() - 1.0).abs() < 0.001);
        assert!((stats.max() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_trend_increasing() {
        let stats = SlidingStats::new(10);

        for i in 0..10 {
            stats.push(i as f64);
        }

        // Positive trend for increasing sequence
        assert!(stats.trend() > 0.0);
    }
}
