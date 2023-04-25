// Copyright 2021 Twitter, Inc.
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

use crate::datapool::*;
use crate::eviction::*;
use crate::item::*;
use crate::segments::*;

use core::num::NonZeroU32;
use rustcommon_time::CoarseInstant as Instant;


/// `Segments` contain all items within the cache. This struct is a collection
/// of individual `Segment`s which are represented by a `SegmentHeader` and a
/// subslice of bytes from a contiguous heap allocation.
pub(crate) struct Segments {
    /// Pointer to slice of headers
    headers: Box<[SegmentHeader]>,
    /// Pointer to raw data
    data: Box<dyn Datapool>,
    /// Segment size in bytes
    segment_size: i32,
    /// Number of free segments
    free: u32,
    /// Total number of segments
    cap: u32,
    /// Head of the free segment queue
    free_q: Option<NonZeroU32>,
    /// Time last flushed
    flush_at: CoarseInstant,
    /// Eviction configuration and state
    evict: Box<Eviction>,
    start_idx: u32,
}

impl Segments {
    /// Private function which allocates and initializes the `Segments` by
    /// taking ownership of the builder
    pub(super) fn from_builder(builder: SegmentsBuilder) -> Self {
        let segment_size = builder.segment_size;
        let segments = builder.heap_size / (builder.segment_size as usize);

        debug!(
            "heap size: {} seg size: {} segments: {}",
            builder.heap_size, segment_size, segments
        );

        assert!(
            segments < (1 << 24), // we use just 24 bits to store the seg id
            "heap size requires too many segments, reduce heap size or increase segment size"
        );

        let evict_policy = builder.evict_policy;

        debug!("eviction policy: {:?}", evict_policy);

        let mut headers = Vec::with_capacity(0);
        headers.reserve_exact(segments);
        for id in 0..segments {
            // safety: we start iterating from 1 and seg id is constrained to < 2^24
            let header = SegmentHeader::new(unsafe { NonZeroU32::new_unchecked(id as u32 + 1 + builder.start_idx) });
            headers.push(header);
        }
        let mut headers = headers.into_boxed_slice();

        let heap_size = segments * segment_size as usize;

        // TODO(bmartin): we always prefault, this should be configurable
        let mut data: Box<dyn Datapool> = if let Some(file) = builder.datapool_path {
            info!("Allocated file backed storage: {:?}", file.as_path());
            let pool = File::create(file, heap_size, true)
                .expect("failed to allocate file backed storage");
            Box::new(pool)
        } else {
            Box::new(Memory::create(heap_size, true))
        };

        for idx in 0..segments {
            let begin = segment_size as usize * idx;
            let end = begin + segment_size as usize;

            let mut segment =
                Segment::from_raw_parts(&mut headers[idx], &mut data.as_mut_slice()[begin..end]);
            segment.init();

            // let id = idx as u32 + 1; // we index segments from 1
            let id = idx as u32 + 1 + builder.start_idx;
            segment.set_prev_seg(NonZeroU32::new(id - 1));
            if id < segments as u32 + builder.start_idx {
                segment.set_next_seg(NonZeroU32::new(id + 1));
            }
        }

        info!("Heap size (byte): {}", heap_size);
        info!("Evict policy: {:?}", evict_policy);
        info!("Start index: {}", builder.start_idx);
        info!("Number of segments: {}", segments);

        Self {
            headers,
            segment_size,
            cap: segments as u32,
            free: segments as u32,
            free_q: NonZeroU32::new(1 + builder.start_idx),
            data,
            flush_at: Instant::recent(),
            evict: Box::new(Eviction::new(segments, evict_policy, builder.start_idx)),
            start_idx: builder.start_idx,
        }
    }

    /// Return the size of each segment in bytes
    #[inline]
    pub fn segment_size(&self) -> i32 {
        self.segment_size
    }

    /// Returns the number of free segments
    #[cfg(test)]
    pub fn free(&self) -> usize {
        self.free as usize
    }

    /// Returns the time the segments were last flushed
    pub fn flush_at(&self) -> CoarseInstant {
        self.flush_at
    }

    /// check if the given segment_id is valid in this Segments
    pub fn seg_id_valid(&self, seg_id: NonZeroU32) -> bool {
        (seg_id.get() > self.start_idx) &&
        (seg_id.get() <= self.cap + self.start_idx)
    }

    /// check if it's DRAM_Segments by looking at start_index
    #[inline]
    pub fn is_dram(&self) -> bool {
        self.start_idx == 0
    }    

    pub fn get_capacity(&self) -> u32 {
        self.cap
    }

    /// Retrieve a `RawItem` from the segment id and offset encoded in the
    /// item info.
    pub(crate) fn get_item(&mut self, item_info: u64) -> Option<RawItem> {
        let seg_id = get_seg_id(item_info);
        let offset = get_offset(item_info) as usize;
        self.get_item_at(seg_id, offset)
    }

    /// Retrieve a `RawItem` from a specific segment id at the given offset
    // TODO(bmartin): consider changing the return type here and removing asserts?
    pub(crate) fn get_item_at(
        &mut self,
        seg_id: Option<NonZeroU32>,
        offset: usize,
    ) -> Option<RawItem> {
        let seg_id = seg_id.map(|v| v.get())?;
        trace!("getting item from: seg: {} offset: {}", seg_id, offset);

        if !self.seg_id_valid(unsafe { NonZeroU32::new_unchecked(seg_id) }) {
            return None;
        }

        let id_idx = (seg_id - 1 - self.start_idx) as usize;
        let seg_begin = self.segment_size() as usize * id_idx;
        let seg_end = seg_begin + self.segment_size() as usize;
        let mut segment = Segment::from_raw_parts(
            &mut self.headers[id_idx],
            &mut self.data.as_mut_slice()[seg_begin..seg_end],
        );

        segment.get_item_at(offset)
    }

    /// Tries to clear a segment by id
    fn clear_segment(
        &mut self,
        id: NonZeroU32,
        hashtable: &mut HashTable,
        expire: bool,
    ) -> Result<(), ()> {
        assert!(self.seg_id_valid(id));

        let mut segment = self.get_mut(id).unwrap();
        if segment.next_seg().is_none() && !expire {
            Err(())
        } else {
            // TODO(bmartin): this should probably result in an error and not be
            // an assert
            assert!(segment.evictable(), "segment was not evictable");
            segment.set_evictable(false);
            segment.set_accessible(false);
            segment.clear(hashtable, expire);
            Ok(())
        }
    }

    /// Perform eviction based on the configured eviction policy. A success from
    /// this function indicates that a segment was put onto the free queue and
    /// that `pop_free()` should return some segment id.
    /// NOTE: should only be called by the segments2
    pub fn evict(
        &mut self,
        ttl_buckets: &mut TtlBuckets,
        hashtable: &mut HashTable,
    ) -> Result<(), SegmentsError> {
        assert!(!self.is_dram(), "DRAM_Segments should not call evict");

        match self.evict.policy() {
            Policy::Merge { .. } => {
                assert!(false, "Merge policy not supported");
                Err(SegmentsError::NoEvictableSegments)
            }
            Policy::None => Err(SegmentsError::NoEvictableSegments),
            _ => {
                if let Some(id) = self.least_valuable_seg(ttl_buckets) {
                    self.clear_segment(id, hashtable, false)
                        .map_err(|_| SegmentsError::EvictFailure)?;

                    let id_idx = (id.get() - 1 - self.start_idx) as usize;
                    // relink head
                    if self.headers[id_idx].prev_seg().is_none() {
                        let ttl_bucket = ttl_buckets.get_mut_bucket(self.headers[id_idx].ttl());
                        ttl_bucket.set_head2(self.headers[id_idx].next_seg());
                    }
                    // relink tail
                    if self.headers[id_idx].next_seg().is_none() {
                        let ttl_bucket = ttl_buckets.get_mut_bucket(self.headers[id_idx].ttl());
                        ttl_bucket.set_tail2(self.headers[id_idx].prev_seg());
                    }
                    self.push_free(id);
                    Ok(())
                } else {
                    Err(SegmentsError::NoEvictableSegments)
                }
            }
        }
    }

    /// Returns a mutable `Segment` view for the segment with the specified id
    pub(crate) fn get_mut(&mut self, id: NonZeroU32) -> Result<Segment, SegmentsError> {
        assert!(self.seg_id_valid(id));
        let id = (id.get() - 1 - self.start_idx) as usize;

        if id < self.headers.len() {
            let header = self.headers.get_mut(id).unwrap();

            let seg_start = self.segment_size as usize * id;
            let seg_end = self.segment_size as usize * (id + 1);

            let seg_data = &mut self.data.as_mut_slice()[seg_start..seg_end];

            let segment = Segment::from_raw_parts(header, seg_data);
            segment.check_magic();
            Ok(segment)
        } else {
            Err(SegmentsError::BadSegmentId)
        }
    }

    /// Helper function which unlinks a segment from a chain by updating the
    /// pointers of previous and next segments.
    /// *NOTE*: this function must not be used on segments in the free queue
    fn unlink(&mut self, id: NonZeroU32) {
        assert!(self.seg_id_valid(id));
        let id_idx = (id.get() - 1 - self.start_idx) as usize;

        if let Some(next) = self.headers[id_idx].next_seg() {
            let prev = self.headers[id_idx].prev_seg();
            self.headers[(next.get() - 1 - self.start_idx) as usize].set_prev_seg(prev);
        }

        if let Some(prev) = self.headers[id_idx].prev_seg() {
            let next = self.headers[id_idx].next_seg();
            self.headers[(prev.get() - 1 - self.start_idx) as usize].set_next_seg(next);
        }
    }

    /// Helper function which pushes a segment onto the front of a chain.
    fn push_front(&mut self, this: NonZeroU32, head: Option<NonZeroU32>) {
        assert!(self.seg_id_valid(this));
        let this_idx = (this.get() - 1 - self.start_idx) as usize;

        self.headers[this_idx].set_next_seg(head);
        self.headers[this_idx].set_prev_seg(None);

        if let Some(head_id) = head {
            assert!(self.seg_id_valid(head_id));
            let head_idx = (head_id.get() - 1 - self.start_idx) as usize;
            debug_assert!(self.headers[head_idx].prev_seg().is_none());
            self.headers[head_idx].set_prev_seg(Some(this));
        }
    }

    /// Returns a segment to the free queue, to be used after clearing the
    /// segment.
    pub(crate) fn push_free(&mut self, id: NonZeroU32) {
        // unlinks the next segment
        self.unlink(id);

        // relinks it as the free queue head
        self.push_front(id, self.free_q);
        self.free_q = Some(id);

        let id_idx = (id.get() - 1 - self.start_idx) as usize;
        assert!(!self.headers[id_idx].evictable());
        self.headers[id_idx].set_accessible(false);

        self.headers[id_idx].reset();

        self.free += 1;
    }

    /// Try to take a segment from the free queue. Returns the segment id which
    /// must then be linked into a segment chain.
    pub(crate) fn pop_free(&mut self) -> Option<NonZeroU32> {
        assert!(self.free <= self.cap);


        if self.free == 0 {
            None
        } else {
            self.free -= 1;
            let id = self.free_q;
            assert!(id.is_some());

            let id_idx = (id.unwrap().get() - 1 - self.start_idx) as usize;

            if let Some(next) = self.headers[id_idx].next_seg() {
                self.free_q = Some(next);
                // this is not really necessary
                let next = &mut self.headers[(next.get() - 1 - self.start_idx) as usize];
                next.set_prev_seg(None);
            } else {
                self.free_q = None;
            }

            #[cfg(not(feature = "magic"))]
            assert_eq!(self.headers[id_idx].write_offset(), 0);

            #[cfg(feature = "magic")]
            assert_eq!(
                self.headers[id_idx].write_offset() as usize,
                std::mem::size_of_val(&SEG_MAGIC),
                "segment: ({}) in free queue has write_offset: ({})",
                id.unwrap(),
                self.headers[id_idx].write_offset()
            );

            // rustcommon_time::refresh_clock();
            self.headers[id_idx].mark_created();
            self.headers[id_idx].mark_merged();

            id
        }
    }

    // TODO(bmartin): use a result here, not option
    /// Returns the least valuable segment based on the configured eviction
    /// policy. An eviction attempt should be made for the corresponding segment
    /// before moving on to the next least valuable segment.
    pub(crate) fn least_valuable_seg(
        &mut self,
        ttl_buckets: &mut TtlBuckets,
    ) -> Option<NonZeroU32> {
        match self.evict.policy() {
            Policy::None => None,
            Policy::Random => {
                let mut start: u32 = self.evict.random();

                start %= self.cap;

                for i in 0..self.cap {
                    let idx = (start + i) % self.cap;
                    if self.headers[idx as usize].can_evict() {
                        // safety: we are always adding 1 to the index
                        return Some(unsafe { NonZeroU32::new_unchecked(idx + 1 + self.start_idx) });
                    }
                }

                None
            }
            Policy::RandomFifo => {
                // This strategy is implemented by picking a random accessible
                // segment and looking up the head of the corresponding
                // `TtlBucket` and evicting that segment. This is functionally
                // equivalent to picking a `TtlBucket` from a weighted
                // distribution based on the number of segments per bucket.

                let mut start: u32 = self.evict.random();

                start %= self.cap;

                for i in 0..self.cap {
                    let idx = (start + i) % self.cap;
                    if self.headers[idx as usize].accessible() {
                        let ttl = self.headers[idx as usize].ttl();
                        let ttl_bucket = ttl_buckets.get_mut_bucket(ttl);
                        // head is the oldest
                        if self.is_dram() {
                            return ttl_bucket.head(); 
                        } else {
                            return ttl_bucket.head2();
                        }
                    }
                }

                None
            }
            // Fifo: first-in-first-out, Cte: closest-to-expiration, Util: least-utilized
            _ => {
                if self.evict.should_rerank() {
                    self.evict.rerank(&self.headers);
                }
                while let Some(id) = self.evict.least_valuable_seg() {
                    if let Ok(seg) = self.get_mut(id) {
                        if seg.can_evict() {
                            return Some(id);
                        }
                    }
                }
                None
            }
        }
    }

    /// Remove a single item from a segment based on the item_info, optionally
    /// setting tombstone
    pub(crate) fn remove_item(
        &mut self,
        item_info: u64,
        tombstone: bool,
        ttl_buckets: &mut TtlBuckets,
        hashtable: &mut HashTable,
    ) -> Result<(), SegmentsError> {
        if let Some(seg_id) = get_seg_id(item_info) {
            let offset = get_offset(item_info) as usize;
            self.remove_at(seg_id, offset, tombstone, ttl_buckets, hashtable)
        } else {
            Err(SegmentsError::BadSegmentId)
        }
    }

    /// Remove a single item from a segment based on the segment id and offset.
    /// Optionally, sets the item tombstone.
    #[allow(unused_variables)]
    pub(crate) fn remove_at(
        &mut self,
        seg_id: NonZeroU32,
        offset: usize,
        tombstone: bool,
        ttl_buckets: &mut TtlBuckets,
        hashtable: &mut HashTable,
    ) -> Result<(), SegmentsError> {
        let is_dram = self.is_dram();

        // remove the item
        {
            let mut segment = self.get_mut(seg_id)?;
            segment.remove_item_at(offset, tombstone);

            // regardless of eviction policy, we can evict the segment if its now
            // empty and would be evictable. if we evict, we must return early
            if segment.live_items() == 0 && segment.can_evict() {
                // NOTE: we skip clearing because we know the segment is empty
                segment.set_evictable(false);
                // if it's the head of a ttl bucket, we need to manually relink
                // the bucket head while we have access to the ttl buckets
                if segment.prev_seg().is_none() {
                    let ttl_bucket = ttl_buckets.get_mut_bucket(segment.ttl());
                    if is_dram  {
                        ttl_bucket.set_head(segment.next_seg()); 
                    } else { 
                        ttl_bucket.set_head2(segment.next_seg());
                    }
                    // manually relink the bucket tail
                    if segment.next_seg().is_none() {
                        let ttl_bucket = ttl_buckets.get_mut_bucket(segment.ttl());
                        if is_dram  { ttl_bucket.set_tail(segment.prev_seg()); } else
                                    { ttl_bucket.set_tail2(segment.prev_seg()); }
                    }
                }
                self.push_free(seg_id);
                return Ok(());
            }
        }

        // we don't do merge for segments2
        if !is_dram {
            return Ok(());
        }

        // for merge eviction, we check if the segment is now below the target
        // ratio which serves as a low watermark for occupancy. if it is, we do
        // a no-evict merge (compaction only, no-pruning)
        if let Policy::Merge { .. } = self.evict.policy() {
            assert!(false, "merge eviction is not supported");
        }

        Ok(())
    }

    // mostly for testing, probably never want to run this otherwise
    #[cfg(any(test, feature = "debug"))]
    pub(crate) fn items(&mut self) -> usize {
        let mut total = 0;
        for id in 1..=self.cap {
            // this is safe because we start iterating from 1
            let segment = self
                .get_mut(unsafe { NonZeroU32::new_unchecked(id as u32) })
                .unwrap();
            segment.check_magic();
            let count = segment.live_items();
            debug!("{} items in segment {} segment: {:?}", count, id, segment);
            total += segment.live_items() as usize;
        }
        total
    }

    #[cfg(test)]
    pub(crate) fn print_headers(&self) {
        for id in 0..self.cap {
            println!("segment header: {:?}", self.headers[id as usize]);
        }
    }

    #[cfg(feature = "debug")]
    pub(crate) fn check_integrity(&mut self) -> bool {
        let mut integrity = true;
        for id in 0..self.cap {
            if !self
                .get_mut(NonZeroU32::new(id + 1).unwrap())
                .unwrap()
                .check_integrity()
            {
                integrity = false;
            }
        }
        integrity
    }


    // NOTE: at the moment this should only be called by segments,
    // move into segments2
    pub fn move_seg_into(
        &mut self,
        seg_id: NonZeroU32,
        ttl_buckets: &mut TtlBuckets,
        hashtable: &mut HashTable, 
        other_segs: &mut Segments,
    ) -> Result<(), SegmentsError> {
        assert!(self.is_dram(), "cannot promote from segments2 yet");
        assert!(!other_segs.is_dram(), "the other segments has to be segments2");

        // 1. input the least/most valuable from the current segment
        assert!(self.seg_id_valid(seg_id));
        let mut old_seg = self.get_mut(seg_id)?;

        // 2. copy the segment content into the new segment
        // MY-LATER-TODO: 
        // current assumption: segment size equals to each other
        // if not enough, we need to use more than one free from the src segments
        let ttl = old_seg.ttl();
        let ttl_bucket = ttl_buckets.get_mut_bucket(ttl);
        loop {
            match ttl_bucket.try_expand(other_segs) {
                Ok(()) => {
                    // expand segments successful, copy into tail
                    if let Some(id) =  ttl_bucket.tail2() {
                        if let Ok(mut tail_seg) = other_segs.get_mut(id) {
                            if !tail_seg.accessible() {
                                continue;
                            }
                            // copy
                            old_seg.copy_into(&mut tail_seg, hashtable)?;
                            // remove the old-segment from ttl_bucket
                            old_seg.clear(hashtable, false);
                            // reset ttl-head
                            if old_seg.prev_seg().is_none() {
                                ttl_bucket.set_head(old_seg.next_seg());
                            }
                            // reset ttl-tail
                            if old_seg.next_seg().is_none() {
                                ttl_bucket.set_tail(old_seg.prev_seg());
                            }
                            self.push_free(seg_id);
                            return Ok(());
                        }
                    }
                },
                Err(_) => {
                    return Err(SegmentsError::NoEvictableSegments);
                }
            }
        }
    }

    /// Try to demote `count` number of segments from seg1 to seg2
    /// Should be **only** called by DRAM_Segments
    pub fn demote(
        &mut self, 
        ttl_buckets: &mut TtlBuckets,
        hashtable: &mut HashTable,
        segments2: &mut Segments,
        count: usize,
    ) -> usize {
        assert!(self.is_dram());
        assert!(!segments2.is_dram());

        match self.evict.policy() {
            Policy::None => 0,
            Policy::Merge { .. } => {
                assert!(false, "Merge demote not suported");
                0
            },
            _ => {
                let mut demoted = 0;

                for _ in 0..count {
                    // select most valuable segment
                    let most_valuable_seg = self.least_valuable_seg(ttl_buckets);
                    if let Some(seg_id) = most_valuable_seg {
                        if self.move_seg_into(
                            seg_id, 
                            ttl_buckets,
                            hashtable,
                            segments2
                        ).is_ok() {
                            // copy into the tail segment of ttl_bucket
                            demoted += 1;
                        } else {
                            for _ in 0..count {
                                let _ = segments2.evict(ttl_buckets, hashtable);
                            }
                        }
                    }            
                }
                // println!("demote {} segments in {:?}", demoted, now.elapsed().as_micros());
                demoted
            }
        }
    }
}

impl Default for Segments {
    fn default() -> Self {
        Self::from_builder(Default::default())
    }
}
