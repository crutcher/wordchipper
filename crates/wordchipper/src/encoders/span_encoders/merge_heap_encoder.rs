//! # Merge Heap based [`SpanEncoder`].
//!
//! Uses a min-heap with lazy deletion and a doubly-linked list for O(m log n)
//! BPE merging, where m is the number of merges and n is the initial token count.

use core::cmp::Reverse;

use crate::{
    TokenType,
    alloc::{collections::BinaryHeap, vec::Vec},
    encoders::span_encoders::span_encoder::SpanEncoder,
    vocab::UnifiedTokenVocab,
};

/// A heap entry: (merge_rank, position, generation_at_push_time).
///
/// Wrapped in [`Reverse`] so the [`BinaryHeap`] acts as a min-heap by rank,
/// with ties broken by position (leftmost first).
type HeapEntry<T> = Reverse<(T, usize, u32)>;

/// A [`SpanEncoder`] using a true min-heap algorithm.
///
/// Uses a [`BinaryHeap`] to find the lowest-rank merge in O(log n) and a
/// doubly-linked list (via index arrays) for O(1) token removal. Stale heap
/// entries are detected via per-position generation counters.
///
/// Working buffers are reused across calls to avoid repeated allocation.
#[derive(Default, Debug, Clone)]
pub struct MergeHeapSpanEncoder<T: TokenType> {
    work_tokens: Vec<T>,
    next: Vec<usize>,
    prev: Vec<usize>,
    generation: Vec<u32>,
    heap: BinaryHeap<HeapEntry<T>>,
}

impl<T: TokenType> SpanEncoder<T> for MergeHeapSpanEncoder<T> {
    fn encode_append_compound_span(
        &mut self,
        vocab: &UnifiedTokenVocab<T>,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        // 1. Build initial byte-level tokens into our working buffer.
        self.work_tokens.clear();
        vocab.byte_vocab().append_tokens(span, &mut self.work_tokens);

        let n = self.work_tokens.len();
        if n <= 1 {
            tokens.extend_from_slice(&self.work_tokens);
            return;
        }

        // 2. Initialize linked-list arrays and generation counters.
        // We use n as the sentinel for "no neighbor".
        let sentinel = n;

        self.next.clear();
        self.next.extend(1..=n); // next[i] = i+1; next[n-1] = n (sentinel)

        self.prev.clear();
        self.prev.reserve(n);
        if n > 0 {
            self.prev.push(sentinel); // prev[0] = sentinel
        }
        self.prev.extend(0..n - 1); // prev[i] = i-1 for i > 0

        self.generation.clear();
        self.generation.resize(n, 0);

        // 3. Seed the heap with all adjacent pairs.
        self.heap.clear();
        let mut pos = 0;
        while self.next[pos] != sentinel {
            let j = self.next[pos];
            if let Some(rank) = vocab.lookup_pair(&(self.work_tokens[pos], self.work_tokens[j])) {
                self.heap.push(Reverse((rank, pos, 0)));
            }
            pos = j;
        }

        // 4. Merge loop.
        while let Some(Reverse((rank, i, entry_gen))) = self.heap.pop() {
            // Skip stale entries.
            if entry_gen != self.generation[i] {
                continue;
            }
            let j = self.next[i];
            if j == sentinel {
                continue;
            }

            // Merge: replace token at i with the merged token, remove j.
            self.work_tokens[i] = rank;

            // Remove j from linked list.
            let k = self.next[j];
            self.next[i] = k;
            if k != sentinel {
                self.prev[k] = i;
            }
            // Mark j as removed by pointing to sentinel.
            self.next[j] = sentinel;

            // Bump generation at i to invalidate any old heap entries for i.
            self.generation[i] = self.generation[i].wrapping_add(1);

            // Recompute pair (prev[i], i) if prev[i] exists.
            let p = self.prev[i];
            if p != sentinel {
                self.generation[p] = self.generation[p].wrapping_add(1);
                if let Some(new_rank) =
                    vocab.lookup_pair(&(self.work_tokens[p], self.work_tokens[i]))
                {
                    self.heap.push(Reverse((new_rank, p, self.generation[p])));
                }
            }

            // Recompute pair (i, next[i]) if next[i] exists.
            if k != sentinel {
                if let Some(new_rank) =
                    vocab.lookup_pair(&(self.work_tokens[i], self.work_tokens[k]))
                {
                    self.heap.push(Reverse((new_rank, i, self.generation[i])));
                }
            }
        }

        // 5. Collect live tokens by walking the linked list.
        let mut pos = 0;
        loop {
            tokens.push(self.work_tokens[pos]);
            let nxt = self.next[pos];
            if nxt == sentinel {
                break;
            }
            pos = nxt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TokenType,
        alloc::{boxed::Box, sync::Arc},
        encoders::{
            span_encoders::TokenSpanEncoder,
            testing::{common_encoder_test_vocab, common_encoder_tests},
        },
        spanning::TextSpannerBuilder,
    };

    fn test_encoder<T: TokenType>() {
        let vocab: Arc<UnifiedTokenVocab<T>> = common_encoder_test_vocab().into();
        let encoder = TokenSpanEncoder::<T>::new(
            TextSpannerBuilder::default(&vocab),
            vocab.clone(),
            Arc::new(|| Box::new(MergeHeapSpanEncoder::<T>::default())),
        );
        common_encoder_tests(vocab.into(), encoder)
    }

    #[test]
    fn test_encoder_u16() {
        test_encoder::<u16>();
    }

    #[test]
    fn test_encoder_u32() {
        test_encoder::<u32>();
    }
}
