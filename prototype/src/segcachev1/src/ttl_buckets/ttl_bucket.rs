// Copyright 2021 Twitter, Inc.
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

//! TTL bucket containing a segment chain which stores items with a similar TTL
//! in an ordered fashion.
//!
//! TTL Bucket:
//! ```text
//! ┌──────────────┬──────────────┬─────────────┬──────────────┐
//! │   HEAD SEG   │   TAIL SEG   │     TTL     │     NSEG     │
//! │              │              │             │              │
//! │    32 bit    │    32 bit    │    32 bit   │    32 bit    │
//! ├──────────────┼──────────────┼─────────────┼──────────────┤
//! │  NEXT MERGE  │ HEAD SEG SSD │ TAIL SEG SSD│   NSEG SSD   │
//! │              │              │             │              │
//! │    32 bit    │    32 bit    │    32 bit   │    32 bit    │
//! ├──────────────┼──────────────┴─────────────┴──────────────┤
//! │NEXT MERGE SSD│                  PADDING                  │
//! │              │                                           │
//! │    32 bit    │                   96 bit                  │
//! ├──────────────┴───────────────────────────────────────────┤
//! │                         PADDING                          │
//! │                                                          │
//! │                         128 bit                          │
//! └──────────────────────────────────────────────────────────┘
//! ```

use crate::*;
use core::num::NonZeroU32;

/// Each ttl bucket contains a segment chain to store items with a similar TTL
/// in an ordered fashion. The first segment to expire will be the head of the
/// segment chain. This allows us to efficiently scan across the [`TtlBuckets`]
/// and expire segments in an eager fashion.
pub struct TtlBucket {
    head: Option<NonZeroU32>,
    head2: Option<NonZeroU32>,
    tail: Option<NonZeroU32>,
    tail2: Option<NonZeroU32>,
    ttl: i32,
    nseg: i32,
    nseg2: i32,
    next_to_merge: Option<NonZeroU32>,
    _pad: [u8; 32],
}

impl TtlBucket {
    pub(super) fn new(ttl: i32) -> Self {
        Self {
            head: None,
            head2: None,
            tail: None,
            tail2: None,
            ttl,
            nseg: 0,
            nseg2: 0,
            next_to_merge: None,
            _pad: [0; 32],
        }
    }

    pub fn head(&self) -> Option<NonZeroU32> {
        self.head
    }

    pub fn head2(&self) -> Option<NonZeroU32> {
        self.head2
    }

    pub fn set_head(&mut self, id: Option<NonZeroU32>) {
        self.head = id;
    }

    pub fn set_head2(&mut self, id: Option<NonZeroU32>) {
        self.head2 = id;
    }

    #[allow(dead_code)]
    pub fn tail(&self) -> Option<NonZeroU32> {
        self.tail
    }

    pub fn tail2(&self) -> Option<NonZeroU32> {
        self.tail2
    }

    pub fn set_tail(&mut self, id: Option<NonZeroU32>) {
        self.tail = id;
    }

    pub fn set_tail2(&mut self, id: Option<NonZeroU32>) {
        self.tail2 = id;
    }

    #[allow(dead_code)]
    pub fn next_to_merge(&self) -> Option<NonZeroU32> {
        self.next_to_merge
    }

    #[allow(dead_code)]
    pub fn set_next_to_merge(&mut self, next: Option<NonZeroU32>) {
        self.next_to_merge = next;
    }

    // expire segments from this TtlBucket, returns the number of segments expired
    pub(super) fn expire(&mut self, hashtable: &mut HashTable, segments: &mut Segments) -> usize {
        let is_dram = segments.is_dram();
        if is_dram {
            if self.head.is_none() { return 0; }
        } else {
            if self.head2.is_none() { return 0; }
        }

        let mut expired = 0;

        loop {
            let seg_id = if is_dram {self.head} else {self.head2};
            if let Some(seg_id) = seg_id {
                let flush_at = segments.flush_at();
                let mut segment = segments.get_mut(seg_id).unwrap();
                if segment.create_at() + segment.ttl() <= CoarseInstant::recent()
                    || segment.create_at() < flush_at
                {
                    if let Some(next) = segment.next_seg() {
                        if is_dram {
                            self.head = Some(next)
                        } else {
                            self.head2 = Some(next)
                        };
                    } else {
                        if is_dram {
                            self.head = None;
                            self.tail = None;
                        } else {
                            self.head2 = None;
                            self.tail2 = None;
                        }
                    }
                    let _ = segment.clear(hashtable, true);
                    segments.push_free(seg_id);
                    expired += 1;
                } else {
                    return expired;
                }
            } else {
                return expired;
            }
        }
    }

    pub(crate) fn try_expand(&mut self, segments: &mut Segments) -> Result<(), TtlBucketsError> {
        let is_dram = segments.is_dram();
        if let Some(id) = segments.pop_free() {
            {
                let ttl_tail = if is_dram {self.tail} else {self.tail2};
                // set (current.tail).next = pop
                if let Some(tail_id) = ttl_tail {
                    let mut tail = segments.get_mut(tail_id).unwrap();
                    tail.set_next_seg(Some(id));
                }
            }

            let mut segment = segments.get_mut(id).unwrap();
            // set pop.prev = current.tail
            segment.set_prev_seg(if is_dram {self.tail} else {self.tail2});
            // set pop.next = None
            segment.set_next_seg(None);
            segment.set_ttl(CoarseDuration::from_secs(self.ttl as u32));

            // init ttl_bucket chain (if needed)
            if is_dram {
                if self.head.is_none() {
                    debug_assert!(self.tail.is_none());
                    self.head = Some(id);
                }
                self.tail = Some(id);
                self.nseg += 1;
            } else {
                if self.head2.is_none() {
                    debug_assert!(self.tail2.is_none());
                    self.head2 = Some(id);
                }
                self.tail2 = Some(id);
                self.nseg2 += 1;
            }

            debug_assert!(!segment.evictable(), "segment should not be evictable");
            segment.set_evictable(true);
            segment.set_accessible(true);
            Ok(())
        } else {
            Err(TtlBucketsError::NoFreeSegments)
        }
    }

    pub(crate) fn reserve(
        &mut self,
        size: usize,
        segments: &mut Segments,
    ) -> Result<ReservedItem, TtlBucketsError> {
        trace!("reserving: {} bytes for ttl: {}", size, self.ttl);

        let seg_size = segments.segment_size() as usize;

        if size > seg_size {
            debug!("item is oversized");
            return Err(TtlBucketsError::ItemOversized { size });
        }

        loop {
            let ttl_tail = if segments.is_dram() {self.tail} else {self.tail2};
            // try to reserve space in current.tail
            if let Some(id) = ttl_tail {
                if let Ok(mut segment) = segments.get_mut(id) {
                    if !segment.accessible() {
                        continue;
                    }
                    let offset = segment.write_offset() as usize;
                    trace!("offset: {}", offset);
                    if offset + size <= seg_size {
                        let size = size as i32;
                        let item = segment.alloc_item(size);
                        return Ok(ReservedItem::new(item, segment.id(), offset));
                    }
                }
            }
            self.try_expand(segments)?;
        }
    }
}
