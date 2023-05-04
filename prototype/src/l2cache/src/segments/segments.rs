// Copyright 2021 Twitter, Inc.
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0

// use ::rand::Rng;

use crate::datapool::*;
use crate::eviction::*;
use crate::item::*;
use crate::segments::*;
use crate::ttl_buckets::*;

use std::collections::HashMap; 
use core::num::NonZeroU32;
use rustcommon_time::CoarseInstant as Instant;

#[cfg(feature="merge_debug1")]
use smallvec::{SmallVec};


/// `Segments` contain all items within the cache. This struct is a collection
/// of individual `Segment`s which are represented by a `SegmentHeader` and a
/// subslice of bytes from a contiguous heap allocation.
pub(crate) struct Segments {
    /// Pointer to slice of headers
    pub headers: Box<[SegmentHeader]>,
    /// Pointer to raw data
    data: Box<dyn Datapool>,
    /// Segment size in bytes
    segment_size: i32,
    /// Number of free segments
    pub free: u32,
    /// Total number of segments
    pub cap: u32,
    /// Head of the free segment queue
    /// TODO: consider remove Option as it uses 8B of memory
    free_q: Option<NonZeroU32>,
    /// Time last flushed
    flush_at: CoarseInstant,
    /// Eviction configuration and state
    pub evict: Box<Eviction>,
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

        // JASONQ: can we just use with_capacity? Reserve_exact may reserve more than need 
        let mut headers = Vec::with_capacity(0);
        headers.reserve_exact(segments);
        for id in 0..segments {
            // safety: we start iterating from 1 and seg id is constrained to < 2^24
            let header = SegmentHeader::new(unsafe { NonZeroU32::new_unchecked(id as u32 + 1) });
            headers.push(header);
        }
        let mut headers = headers.into_boxed_slice();

        let heap_size = segments * segment_size as usize;

        // TODO(bmartin): we always prefault, this should be configurable
        let mut data: Box<dyn Datapool> = if let Some(file) = builder.datapool_path {
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
            debug_assert!(segment.header.is_free()); 

            let id = idx as u32 + 1; // we index segments from 1
            segment.set_prev_seg(NonZeroU32::new(id - 1));
            if id < segments as u32 {
                segment.set_next_seg(NonZeroU32::new(id + 1));
            }
        }

        Self {
            headers,
            segment_size,
            cap: segments as u32,
            free: segments as u32,
            free_q: NonZeroU32::new(1),
            data,
            flush_at: Instant::recent(),
            evict: Box::new(Eviction::new(segments, evict_policy)),
        }
    }

    /// Return the size of each segment in bytes
    #[inline]
    pub fn segment_size(&self) -> i32 {
        self.segment_size
    }

    #[inline]
    #[allow(dead_code)]
    pub fn get_cache_size(&self) -> usize {
        self.segment_size as usize * self.cap as usize 
    }

    /// Returns the number of free segments
    #[cfg(test)]
    pub fn free(&self) -> usize {
        self.free as usize
    }

    /// Returns the time the segments were last flushed
    #[inline]
    #[allow(dead_code)]
    pub fn flush_at(&self) -> CoarseInstant {
        self.flush_at
    }

    #[inline]
    #[allow(dead_code)]
    pub fn n_free(&self) -> u32 {
        self.free
    }

    #[inline]
    #[allow(dead_code)]
    pub fn get_target_size(&self) -> i32 {
        (self.segment_size() as f64 * self.evict.target_ratio()) as i32
    }

    #[allow(dead_code)]
    pub fn cal_mean_utilization(&self) ->f64 {
        let mut total_size = 0;
        let mut total_used = 0;
        for id in 0..self.cap {
            let header = &self.headers[id as usize];
            if !header.is_free() {
                total_size += self.segment_size as usize;
                total_used += header.live_bytes() as usize;
            }
        }

        total_used as f64 / total_size as f64
    }

    #[allow(dead_code)]
    pub fn print_mean_utilization(&self) {
        let mut total_size = 0;
        let mut total_used = 0;
        for id in 0..self.cap {
            let header = &self.headers[id as usize];
            total_size += self.segment_size as usize;
            if !header.is_free() {
                // total_used += header.live_bytes() as usize;
                total_used += header.live_bytes() as usize;
            }
        }

        total_size = total_size / (1024 * 1024);
        total_used = total_used / (1024 * 1024);
        println!("free/cap {}/{} mean util {}/{} = {:.4}", self.free, self.cap, total_used, total_size, total_used as f64 / total_size as f64);
    }

    /// unlink a segment from the segment chain 
    /// 
    #[inline]
    pub fn unlink_segment(&mut self, id: NonZeroU32, ttl_bucket: Option<&mut TtlBucket>) {
        // self.verify_global_segment_chain(); 

        let id_idx = id.get() as usize - 1;

        if let Some(ttl_bucket) = ttl_bucket {
            if !ttl_bucket.head().is_none() && ttl_bucket.head().unwrap() == id {
                ttl_bucket.set_head(self.headers[id_idx].next_seg());
            }

            if !ttl_bucket.tail().is_none() && ttl_bucket.tail().unwrap() == id {
                ttl_bucket.set_tail(self.headers[id_idx].prev_seg());
            }

            if !ttl_bucket.seg_before_tail.is_none() && ttl_bucket.seg_before_tail.unwrap() == id {
                ttl_bucket.seg_before_tail = self.headers[id_idx].prev_seg();
            }

            if !ttl_bucket.next_to_merge().is_none() && ttl_bucket.next_to_merge().unwrap() == id {
                ttl_bucket.set_next_to_merge(self.headers[id_idx].next_seg());
            }

            ttl_bucket.reduce_nseg(1); 
        }

        // unlink from the TTL segment chain 
        let prev_id = self.headers[id_idx].prev_seg();
        let next_id = self.headers[id_idx].next_seg();

        if let Some(prev_id) = prev_id {
            let prev_id = prev_id.get();
            self.headers[prev_id as usize - 1].set_next_seg(next_id);
        } 

        if let Some(next_id) = next_id {
            let next_id = next_id.get();
            self.headers[next_id as usize - 1].set_prev_seg(prev_id);
        } 
    }


    // #[inline]
    // #[allow(dead_code)]
    // pub(crate) fn reduce_live_bytes(&mut self, seg_id: u32, size: i32) {
    //     self.headers[seg_id as usize].decr_live_bytes(size);
    // }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn increase_live_bytes(&mut self, seg_id: u32, size: i32) {
        self.headers[seg_id as usize].incr_live_bytes(size);
    }

    #[allow(dead_code)]
    pub(crate) fn verify_segment_integrity(&mut self, id: NonZeroU32, hashtable: &mut HashTable) {
        let mut segment = self.get_mut(id).unwrap(); 
        segment.verify_integrity(hashtable);
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn get_age(&self, item_info: u64) -> i32 {
        let seg_id = get_seg_id(item_info);
        
        self.headers[seg_id.unwrap().get() as usize - 1].create_at().elapsed().as_secs() as i32
    }

    /// Retrieve a `RawItem` from the segment id and offset encoded in the
    /// item info.
    #[inline]
    pub(crate) fn get_item(&mut self, item_info: u64) -> RawItem {
        let seg_id = get_seg_id(item_info);
        let offset = get_offset(item_info) as usize;
        self.get_item_at(seg_id, offset)
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn unchecked_get_item_at(
        &mut self,
        seg_id: Option<NonZeroU32>,
        offset: usize,
    ) -> RawItem {
        let seg_id = seg_id.unwrap().get();
        debug_assert!(seg_id <= self.cap as u32);

        let seg_begin = self.segment_size() as usize * (seg_id as usize - 1);
        let seg_end = seg_begin + self.segment_size() as usize;
        let mut segment = Segment::from_raw_parts(
            &mut self.headers[seg_id as usize - 1],
            &mut self.data.as_mut_slice()[seg_begin..seg_end],
        );

        segment.unchecked_get_item_at(offset) 
    }

    /// Retrieve a `RawItem` from a specific segment id at the given offset
    pub(crate) fn get_item_at(
        &mut self,
        seg_id: Option<NonZeroU32>,
        offset: usize,
    ) -> RawItem {
        let seg_id = seg_id.unwrap().get();
        trace!("getting item from: seg: {} offset: {}", seg_id, offset);
        debug_assert!(seg_id <= self.cap as u32);

        let seg_begin = self.segment_size() as usize * (seg_id as usize - 1);
        let seg_end = seg_begin + self.segment_size() as usize;
        let mut segment = Segment::from_raw_parts(
            &mut self.headers[seg_id as usize - 1],
            &mut self.data.as_mut_slice()[seg_begin..seg_end],
        );

        segment.get_item_at(offset) 
    }

    /// calculate segmente utility using oracle info
    #[cfg(feature="oracle_reuse")]
    pub(crate) fn segment_utility(&mut self, seg_idx: usize, curr_vtime: u64, retain_frac: f32) ->f32 {
        let seg_begin = self.segment_size() as usize * seg_idx;
        let seg_end = seg_begin + self.segment_size() as usize;
        let mut segment = Segment::from_raw_parts(
            &mut self.headers[seg_idx],
            &mut self.data.as_mut_slice()[seg_begin..seg_end],
        );

        let mut utility_vec = Vec::<f32>::with_capacity(segment.live_items() as usize);
        let max_offset = segment.max_item_offset();
        let mut offset = segment.get_offset_start(); 

        while offset <= max_offset {
            let item = segment.get_item_at(offset);
            if item.klen() == 0 && segment.live_items() == 0 {
                break;
            }

            let item_size = item.size();
            if item.is_deleted() {
                offset += item_size;
                continue;
            }

            let future_reuse_vtime = item.header().get_future_reuse_time(); 
            let dist = (future_reuse_vtime as u64 - curr_vtime);
            let size = item_size;
            let utility = 1.0e10 / size as f32 / dist as f32;
            utility_vec.push(utility);
            offset += item_size;
        }

        let n_retained = (segment.live_items() as f32 * retain_frac) as usize;
        if n_retained > 0 {
            utility_vec.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        }

        utility_vec[n_retained..].iter().sum::<f32>()
    }

    #[cfg(feature="oracle_reuse")]
    pub(crate) fn update_segment_pred_utility(&mut self, curr_vtime: u64, retain_frac: f32) {
        for seg_idx in 0..self.cap as usize {
            // segment.header.pred_utility = quickrandom() as f32; 
            // segment.header.pred_utility = 1.0e6 / segment.create_at().elapsed().as_secs() as f32;

            self.headers[seg_idx].pred_utility = self.segment_utility(seg_idx, curr_vtime, retain_frac); 
            self.headers[seg_idx].can_evict_this_round = true; 
        }
    }

    #[cfg(feature="oracle_reuse")]
    #[allow(dead_code)]
    pub(crate) fn cal_offline_segment_utility(&mut self, curr_vtime: u64) {
        let n_merge = self.evict.n_merge(); 

        let retain_frac = 1.0 / n_merge as f32;
        for seg_idx in 0..self.cap as usize {
            if self.headers[seg_idx].train_data_idx == -1 {
                continue; 
            }
            
            let utility = self.segment_utility(seg_idx, curr_vtime, retain_frac);

            #[cfg(feature="offline_segment_utility")]
            self.evict.get_l2learner().set_train_segment_utility(
                self.headers[seg_idx].train_data_idx, 
                utility,
            );

            self.evict.get_l2learner().set_offline_segment_utility(
                self.headers[seg_idx].train_data_idx, 
                utility,
            );
        } 
    }

    /// Tries to clear a segment by id, 
    /// force evict can evict the tail segment 
    fn clear_segment(
        &mut self,
        id: NonZeroU32,
        hashtable: &mut HashTable, 
        expire: bool,
        force_evict: bool
    ) -> Result<(), ()> {
        let mut segment = self.get_mut(id).unwrap();
        if segment.next_seg().is_none() && !expire && !force_evict {
            Err(())
        } else {
            // an assert
            assert!(segment.evictable(), "segment was not evictable");
            segment.set_evictable(false);
            segment.set_accessible(false);
            segment.clear(hashtable, expire);
            Ok(())
        }
    }

    /// TODO: currently resets the full cache, but we only need to reset for the sampled segments
    pub(crate) fn reset_accessed_since_snapshot(
        &mut self, 
    ) {
        let segment_size = self.segment_size(); 
        for seg_idx in 0..self.cap as usize {
            let seg_begin = segment_size as usize * seg_idx;
            let seg_end = seg_begin + segment_size as usize;
            let mut segment = Segment::from_raw_parts(
                &mut self.headers[seg_idx],
                &mut self.data.as_mut_slice()[seg_begin..seg_end],
            );

            if segment.header.train_data_idx == -1 || segment.header.free {
                continue; 
            }
            
            let max_offset = segment.max_item_offset();
            let mut offset = segment.get_offset_start(); 
    
            while offset <= max_offset {
                let mut item = segment.get_item_at(offset);
                if item.klen() == 0 && segment.live_items() == 0 {
                    break;
                }
    
                let item_size = item.size();
                if item.is_deleted() {
                    offset += item_size;
                    continue;
                }
            
                item.set_accessed_since_snapshot(false);
                assert!(!item.has_accessed_since_snapshot());
                offset += item_size;
            }
        }
    }

    pub fn evict(
        &mut self,
        ttl_buckets: &mut TtlBuckets,
        hashtable: &mut HashTable,
        curr_vtime: u64,
        ghost_map: &mut HashMap<u64, u64>,
    ) -> Result<(), SegmentsError> { 

        let curr_sec = CoarseInstant::recent().as_secs();
        let should_rerank = self.evict.should_rerank(); 

        match self.evict.policy_mut() {
            Policy::Merge { .. } => {
                return Err(SegmentsError::NoEvictableSegments);
            }
            Policy::OracleMerge { n_merge:_n_merge, .. } => {
                return Err(SegmentsError::NoEvictableSegments);
            }
            Policy::LearnedMerge{train_interval_sec, time_before_first_train_data, learner, ..} => {                
                // use learned eviction when we have a model 
                if curr_sec > *time_before_first_train_data + *train_interval_sec
                    && should_rerank {
                    let mut performed_training = false;

                    if curr_sec >= learner.next_train_time {
                        
                        learner.train(); 

                        ghost_map.clear();

                        // sample segments for training, the y will be calculated over time
                        learner.gen_training_data(&mut self.headers, curr_vtime);
                        learner.next_train_time = curr_sec + *train_interval_sec;

                        performed_training = true;
                    }


                    learner.inference(&mut self.headers); 
                    // self.update_segment_pred_utility(curr_vtime, 0.0); 

                    self.evict.rerank(&self.headers);
                    
                    // needed after each rank 
                    self.headers.iter_mut().for_each(|h| {h.can_evict_this_round = true; }); 

                    if performed_training {
                        self.reset_accessed_since_snapshot();

                        #[cfg(feature="oracle_reuse")] 
                        // calculate offline segment utility
                        self.cal_offline_segment_utility(curr_vtime); 
                    }
                }
                
                let mut num_evicted = 0;
                while num_evicted < 10 {
                    if let Some(id) = self.least_valuable_seg(ttl_buckets) {
                        if let Err(err) = self.clear_segment(id, hashtable, false, false) {
                            debug!("clear err {:?}", err); 
                            return Err(SegmentsError::EvictFailure);
                        }

                        let id_idx = id.get() as usize - 1;
                        let ttl_bucket = ttl_buckets.get_mut_bucket(self.headers[id_idx].ttl());
                        self.push_free(id, Some(ttl_bucket));
                        num_evicted += 1;
                        
                    } else if num_evicted == 0 {
                        return Err(SegmentsError::NoEvictableSegments);
                    } else {
                        return Ok(())
                    }
                }
                return Ok(());

            }
            Policy::None => Err(SegmentsError::NoEvictableSegments),
            _ => {
                if let Some(id) = self.least_valuable_seg(ttl_buckets) {
                    if let Err(err) = self.clear_segment(id, hashtable, false, false) {
                        debug!("clear err {:?}", err); 
                        return Err(SegmentsError::EvictFailure)
                    }

                    let id_idx = id.get() as usize - 1;
                    let ttl_bucket = ttl_buckets.get_mut_bucket(self.headers[id_idx].ttl());
                    self.push_free(id, Some(ttl_bucket));

                    Ok(())
                } else {
                    Err(SegmentsError::NoEvictableSegments)
                }
            }
        }
    }

    /// Returns a mutable `Segment` view for the segment with the specified id
    pub(crate) fn get_mut(&mut self, id: NonZeroU32) -> Result<Segment, SegmentsError> {
        let id = id.get() as usize - 1;
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

    /// Gets a mutable `Segment` view for two segments after making sure the
    /// borrows are disjoint.
    #[allow(dead_code)]
    pub(crate) fn get_mut_pair(
        &mut self,
        a: NonZeroU32,
        b: NonZeroU32,
    ) -> Result<(Segment, Segment), SegmentsError> {
        if a == b {
            Err(SegmentsError::BadSegmentId)
        } else {
            let a = a.get() as usize - 1;
            let b = b.get() as usize - 1;
            if a >= self.headers.len() || b >= self.headers.len() {
                return Err(SegmentsError::BadSegmentId);
            }
            // we have already guaranteed that 'a' and 'b' are not the same, so
            // we know that they are disjoint borrows and can safely return
            // mutable borrows to both the segments
            unsafe {
                let seg_size = self.segment_size() as usize;

                let header_a = &mut self.headers[a] as *mut _;
                let header_b = &mut self.headers[b] as *mut _;

                let data = self.data.as_mut_slice();

                // split the borrowed data
                let split = (std::cmp::min(a, b) + 1) * seg_size;
                let (first, second) = data.split_at_mut(split);

                let (data_a, data_b) = if a < b {
                    let start_a = seg_size * a;
                    let end_a = seg_size * (a + 1);

                    let start_b = (seg_size * b) - first.len();
                    let end_b = (seg_size * (b + 1)) - first.len();

                    (&mut first[start_a..end_a], &mut second[start_b..end_b])
                } else {
                    let start_a = (seg_size * a) - first.len();
                    let end_a = (seg_size * (a + 1)) - first.len();

                    let start_b = seg_size * b;
                    let end_b = seg_size * (b + 1);

                    (&mut second[start_a..end_a], &mut first[start_b..end_b])
                };

                let segment_a = Segment::from_raw_parts(&mut *header_a, data_a);
                let segment_b = Segment::from_raw_parts(&mut *header_b, data_b);

                segment_a.check_magic();
                segment_b.check_magic();
                Ok((segment_a, segment_b))
            }
        }
    }

    /// Helper function which pushes a segment onto the front of a chain.
    fn push_front(&mut self, this: NonZeroU32, head: Option<NonZeroU32>) {
        let this_idx = this.get() as usize - 1;
        self.headers[this_idx].set_next_seg(head);
        self.headers[this_idx].set_prev_seg(None);

        if let Some(head_id) = head {
            let head_idx = head_id.get() as usize - 1;
            debug_assert!(self.headers[head_idx].prev_seg().is_none());
            self.headers[head_idx].set_prev_seg(Some(this));
        }
    }

    /// Returns a segment to the free queue, to be used after clearing the
    /// segment.
    pub(crate) fn push_free(&mut self, id: NonZeroU32, ttl_bucket: Option<&mut TtlBucket>) {
        // unlinks the next segment
        self.unlink_segment(id, ttl_bucket);

        let id_idx = id.get() as usize - 1;

        // relinks it as the free queue head
        self.push_front(id, self.free_q);
        self.free_q = Some(id);

        assert!(!self.headers[id_idx].evictable());
        self.headers[id_idx].set_accessible(false);

        self.headers[id_idx].set_free(); 
        self.free += 1;

        // println!("{} {:?}", self.free, self.free_q.unwrap().get()); 
        debug_assert!(self.free <= 1 || 
            self.headers[self.free_q.unwrap().get() as usize - 1].next_seg().is_some());
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

            let id_idx = id.unwrap().get() as usize - 1;

            if let Some(next) = self.headers[id_idx].next_seg() {
                self.free_q = Some(next);
                // this is not really necessary
                let next = &mut self.headers[next.get() as usize - 1];
                next.set_prev_seg(None);
            } else {
                self.free_q = None;
            }

            assert!(self.headers[id_idx].is_free()); 
            self.headers[id_idx].init();
            self.headers[id_idx].set_not_free(); 

            // #[cfg(not(feature = "magic"))]
            // assert_eq!(self.headers[id_idx].write_offset(), 0);

            // #[cfg(feature = "magic")]
            // assert_eq!(
            //     self.headers[id_idx].write_offset() as usize,
            //     std::mem::size_of_val(&SEG_MAGIC),
            //     "segment: ({}) in free queue has write_offset: ({})",
            //     id.unwrap(),
            //     self.headers[id_idx].write_offset()
            // );

            // rustcommon_time::refresh_clock();
            // self.headers[id_idx].mark_created();
            // self.headers[id_idx].reset_merge_at(); 


            id
        }
    }

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
                let mut start: u32 = (quickrandom() % (u32::MAX as u64)) as u32;

                start %= self.cap;

                for i in 0..self.cap {
                    let idx = (start + i) % self.cap;
                    if self.headers[idx as usize].can_evict() {
                        // safety: we are always adding 1 to the index
                        return Some(unsafe { NonZeroU32::new_unchecked(idx + 1) });
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
                    if self.headers[idx as usize].evictable() {
                        let ttl = self.headers[idx as usize].ttl();
                        let ttl_bucket = ttl_buckets.get_mut_bucket(ttl);
                        return ttl_bucket.head();
                    }
                }

                None
            }
            Policy::LearnedMerge {train_interval_sec, time_before_first_train_data, ..} => {
                if CoarseInstant::recent().as_secs() <= *time_before_first_train_data + *train_interval_sec {
                    let mut start: u32 = (quickrandom() % (u32::MAX as u64)) as u32;

                    start %= self.cap;
    
                    for i in 0..self.cap {
                        let idx = (start + i) % self.cap;
                        if self.headers[idx as usize].evictable() {
                            if self.headers[idx as usize].can_evict() {
                                // safety: we are always adding 1 to the index
                                return Some(unsafe { NonZeroU32::new_unchecked(idx + 1) });
                            }
                        }
                    }
                }

                if self.evict.should_rerank() {
                    self.evict.rerank(&self.headers, );
                }
                while let Some(id) = self.evict.least_valuable_seg() {
                    if let Ok(seg) = self.get_mut(id) {
                        if seg.evictable() {
                            if seg.can_evict() {
                                return Some(id);
                            }
                        }
                    }
                }
                None
            }
            _ => {
                if self.evict.should_rerank() {
                    self.evict.rerank(&self.headers, );
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
        curr_vtime: u64,
    ) -> Result<(), SegmentsError> {
        if let Some(seg_id) = get_seg_id(item_info) {
            let offset = get_offset(item_info) as usize;
            self.remove_at(seg_id, offset, tombstone, ttl_buckets, hashtable, curr_vtime)
        } else {
            Err(SegmentsError::BadSegmentId)
        }
    }

    /// Remove a single item from a segment based on the segment id and offset.
    /// Optionally, sets the item tombstone.
    pub(crate) fn remove_at(
        &mut self,
        seg_id: NonZeroU32,
        offset: usize,
        tombstone: bool,
        ttl_buckets: &mut TtlBuckets,
        _hashtable: &mut HashTable, 
        _curr_vtime: u64, 
    ) -> Result<(), SegmentsError> {
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
                // println!("remove at seg {}", seg_id.get());
                let ttl_bucket = ttl_buckets.get_mut_bucket(segment.ttl());
                self.push_free(seg_id, Some(ttl_bucket));
                return Ok(());
            }
        }

        // if let Policy::Merge { .. } = self.evict.policy() {
        //     let n_compact = self.evict.n_compact();             
        //     let mut id_idx = seg_id.get() as usize - 1;
        //     let ratio = self.headers[id_idx].live_bytes() as f64 / self.segment_size() as f64;

        //     if ratio > 1.0 - 1.0 / n_compact as f64 {
        //         return Ok(());
        //     }

        //     let mut live_ratio_sum = ratio; 
        //     for _i in 1..n_compact {
        //         let next_id = self.headers[id_idx].next_seg();
        //         // println!("{} check {:?} ", _i, next_id); 
        //         if let Some(next_id) = next_id {
        //             id_idx = next_id.get() as usize - 1;
        //             if !self.headers[id_idx].can_evict() {
        //                 return Ok(()); 
        //             }

        //             let next_ratio = self.headers[id_idx].live_bytes() as f64 / self.segment_size() as f64;
        //             live_ratio_sum += next_ratio;
        //         } else {
        //             return Ok(());
        //         }
        //     }

        //     if live_ratio_sum <= n_compact as f64 - 1.0 {
        //         let ttl_bucket = ttl_buckets.get_mut_bucket(self.headers[id_idx].ttl());
        //         let _ = self.merge_compact(seg_id, hashtable, ttl_bucket, curr_vtime);
        //     }
        // }
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

    #[allow(dead_code)]
    pub(crate) fn print_segment_chain(&self, start: Option<NonZeroU32>) {
        let mut id = start;
        while let Some(idx) = id {
            print!("{}, ", idx.get()); 
            id = self.headers[idx.get() as usize - 1].next_seg()
        }

        println!(""); 
    }

    #[allow(dead_code)]
    pub fn get_segment_chain_str(&self, start: Option<NonZeroU32>) -> String {
        let mut s = String::new();
        let mut id = start;
        while let Some(idx) = id {
            s = s + format!("{} ({}), ", idx.get(), self.headers[idx.get() as usize - 1].is_free()).as_str(); 
            id = self.headers[idx.get() as usize - 1].next_seg()
        }

        s
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

    #[allow(dead_code)]
    pub(crate) fn print_segment_chain_unevictable_reason(&self, ttl_bucket: &mut TtlBucket, n: i32) {

        let mut curr_id = ttl_bucket.head().unwrap();
        let mut n_printed = 0;
        loop {
            let id_idx = curr_id.get() - 1; 
            print!("{:?}({}), ", self.headers[id_idx as usize].not_evictable_reason(), id_idx);
            n_printed += 1;
            if n_printed == n {
                break;
            }
            match self.headers[id_idx as usize].next_seg() {
                Some(next_id) => {
                    curr_id = next_id;
                }
                None => {
                    break;
                }
            }
        }
        println!("");
    }

    #[allow(dead_code)]
    pub fn verify_segments_status(&self, msg: &str) {
        let mut n_free = 0;
        let mut n_use = 0; 
        for i in 0..self.headers.len() {
            let header = &self.headers[i];
            if header.is_free() {
                n_free += 1;
            } else {
                n_use += 1;
            }
        }
        assert_eq!(n_free + n_use, self.cap, "{}", msg); 
        assert_eq!(n_free, self.free, "{}", msg);
    }

    #[allow(dead_code)]
    fn verify_segment_chain_order(&self, start_id: Option<NonZeroU32>) {
        let mut opt_id = start_id; 
        let mut last_time = 0; 
        while let Some(id) = opt_id {
            let header = &self.headers[id.get() as usize -1];
            let curr_time = header.create_at().as_secs();
            assert!(curr_time >= last_time);
            last_time = curr_time;
            opt_id = header.next_seg();
        }
    }
        

    #[allow(dead_code)]
    pub fn print_segment_header(&self, id: NonZeroU32) {
        let header = &self.headers[id.get() as usize - 1];
        println!("segment {}: {:?}", id, header);
    }

    #[allow(dead_code)]
    pub fn print_objects(&mut self, seg_idx: usize) {
        let header = &self.headers[seg_idx];
        println!("segment {}: {:?}", seg_idx, header);

        let seg_begin = self.segment_size() as usize * seg_idx;
        let seg_end = seg_begin + self.segment_size() as usize;
        let mut segment = Segment::from_raw_parts(
            &mut self.headers[seg_idx],
            &mut self.data.as_mut_slice()[seg_begin..seg_end],
        );

        let max_offset = segment.max_item_offset();
        let mut offset = segment.get_offset_start(); 

        // let mut n_item = 0; 
        while offset <= max_offset {
            let item = segment.get_item_at(offset);
            if item.klen() == 0 && segment.live_items() == 0 {
                break;
            }

            let item_size = item.size();
            offset += item_size;
        }
    }

    #[allow(dead_code)]
    pub fn is_in_the_free_queue(&self, id: NonZeroU32) -> bool {
        let mut cur = self.free_q;
        while let Some(curr) = cur {
            let curr_id = curr.get();
            if curr_id == id.get() {
                return true;
            }
            cur = self.headers[curr_id as usize - 1].next_seg();
        }

        false
    }

    #[allow(dead_code)]
    pub fn is_prev_in_the_free_queue(&self, id: NonZeroU32) -> bool {
        let prev_id = self.headers[id.get() as usize - 1].prev_seg();

        if let Some(prev_id) = prev_id {
            self.is_in_the_free_queue(prev_id)
        } else {
            false
        }
    }

    #[allow(dead_code)]
    pub fn is_next_in_the_free_queue(&self, id: NonZeroU32) -> bool {
        let next_id = self.headers[id.get() as usize - 1].next_seg();

        if let Some(next_id) = next_id {
            self.is_in_the_free_queue(next_id)
        } else {
            false
        }
    }


}



impl Default for Segments {
    fn default() -> Self {
        Self::from_builder(Default::default())
    }
}
